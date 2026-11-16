"""Drift detection methods: CUSUM, PELT, and LSTM Autoencoder.

Implements sequential changepoint detection and anomaly scoring on monthly
normalized aspect sentiment time series:
1. Two-sided CUSUM sequential control chart (per-aspect with alarm cooldown)
2. PELT changepoint detection via ruptures (nonparametric CostRank)
3. 1-Layer LSTM Autoencoder with latent bottleneck and masked reconstruction loss
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
try:
    import ruptures as rpt
except ImportError:
    rpt = None
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .utils import CANONICAL_ASPECTS, set_seed

__all__ = [
    # CUSUM
    "detect_cusum",
    "detect_cusum_product",
    "cusum_fuse",
    # PELT
    "forward_fill_series",
    "detect_pelt",
    "detect_pelt_product",
    "pelt_fuse",
    # LSTM Autoencoder
    "SentimentWindowDataset",
    "collate_fn",
    "LSTMAutoencoder",
    "masked_mse_loss",
    "train_model",
    "score_anomalies",
    "detect_lstm_ae_product",
]


# ---------------------------------------------------------------------------
# Two-Sided CUSUM Sequential Control Chart
# ---------------------------------------------------------------------------
def detect_cusum(
    series: np.ndarray,
    mask: np.ndarray,
    k: float = 0.5,
    h: float = 5.0,
    cooldown: int = 2,
) -> dict:
    """Run two-sided CUSUM on a single product-aspect time series.

    Parameters
    ----------
    series : 1D array of normalized aspect scores.
    mask : 1D boolean array (True where month is observed).
    k : Allowance (slack) parameter (half of expected shift size).
    h : Decision threshold. Alarm triggers when S+ or S- exceeds h.
    cooldown : Months to skip after alarm before monitoring resumes.

    Returns
    -------
    dict with:
      - alarm_indices: list[int]
      - S_plus: np.ndarray positive accumulator
      - S_minus: np.ndarray negative accumulator
      - alarm_directions: list[str] ('positive' or 'negative')
    """
    T = len(series)
    if T != len(mask):
        raise ValueError("series and mask must have identical length")

    S_plus = np.zeros(T)
    S_minus = np.zeros(T)
    alarm_indices = []
    alarm_directions = []
    cooldown_remaining = 0

    for t in range(T):
        if cooldown_remaining > 0:
            S_plus[t] = 0.0
            S_minus[t] = 0.0
            cooldown_remaining -= 1
            continue

        if not mask[t]:
            # Missing month: carry forward previous accumulator
            S_plus[t] = S_plus[t - 1] if t > 0 else 0.0
            S_minus[t] = S_minus[t - 1] if t > 0 else 0.0
            continue

        x = series[t]
        s_prev_plus = S_plus[t - 1] if t > 0 else 0.0
        s_prev_minus = S_minus[t - 1] if t > 0 else 0.0

        S_plus[t] = max(0.0, s_prev_plus + x - k)
        S_minus[t] = max(0.0, s_prev_minus - x - k)

        if S_plus[t] > h:
            alarm_indices.append(t)
            alarm_directions.append("positive")
            S_plus[t] = 0.0
            S_minus[t] = 0.0
            cooldown_remaining = cooldown
        elif S_minus[t] > h:
            alarm_indices.append(t)
            alarm_directions.append("negative")
            S_plus[t] = 0.0
            S_minus[t] = 0.0
            cooldown_remaining = cooldown

    return {
        "alarm_indices": alarm_indices,
        "S_plus": S_plus,
        "S_minus": S_minus,
        "alarm_directions": alarm_directions,
    }


def detect_cusum_product(
    product_df: pd.DataFrame,
    aspects: list[str],
    k: float = 0.5,
    h: float = 5.0,
    cooldown: int = 2,
    score_col: str = "norm_score",
) -> pd.DataFrame:
    """Run two-sided CUSUM across all aspects of a single product."""
    months = sorted(product_df["year_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months)}

    results = []
    for asp in aspects:
        asp_data = product_df[product_df["canonical_aspect"] == asp].sort_values("year_month")
        if len(asp_data) == 0:
            continue

        series = np.full(len(months), 0.0)
        mask = np.zeros(len(months), dtype=bool)

        for _, row in asp_data.iterrows():
            idx = month_to_idx.get(row["year_month"])
            if idx is not None:
                val = row.get(score_col, row.get("norm_score"))
                series[idx] = val if pd.notna(val) else 0.0
                mask[idx] = bool(row["aspect_mask"])

        cusum_result = detect_cusum(series, mask, k=k, h=h, cooldown=cooldown)
        alarm_set = set(cusum_result["alarm_indices"])
        dir_map = dict(zip(cusum_result["alarm_indices"], cusum_result["alarm_directions"]))

        for i, m in enumerate(months):
            results.append({
                "year_month": m,
                "canonical_aspect": asp,
                "is_changepoint": i in alarm_set,
                "direction": dir_map.get(i, None),
            })

    return pd.DataFrame(results)


def cusum_fuse(detections_df: pd.DataFrame, tolerance: int = 1) -> pd.DataFrame:
    """Merge per-aspect CUSUM alarms within +/- tolerance months into product-level events."""
    cps = detections_df[detections_df["is_changepoint"]].copy()
    if len(cps) == 0:
        return pd.DataFrame(columns=["event_month", "aspects_involved", "n_aspects", "directions"])

    cps = cps.sort_values("year_month").reset_index(drop=True)
    months = sorted(cps["year_month"].unique())

    clusters = []
    current_cluster = [months[0]]
    for m in months[1:]:
        prev = current_cluster[-1]
        diff = (m.year - prev.year) * 12 + (m.month - prev.month)
        if diff <= tolerance:
            current_cluster.append(m)
        else:
            clusters.append(current_cluster)
            current_cluster = [m]
    clusters.append(current_cluster)

    events = []
    for cluster_months in clusters:
        cluster_cps = cps[cps["year_month"].isin(cluster_months)]
        aspects = sorted(cluster_cps["canonical_aspect"].unique())
        directions = sorted(cluster_cps["direction"].dropna().unique())
        events.append({
            "event_month": min(cluster_months),
            "aspects_involved": aspects,
            "n_aspects": len(aspects),
            "directions": directions,
        })
    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# PELT Changepoint Detection via ruptures
# ---------------------------------------------------------------------------
def forward_fill_series(series: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last Observation Carried Forward (LOCF) for missing time series values."""
    filled = series.copy()
    filled_mask = mask.copy()
    last_val = np.nan
    for i in range(len(filled)):
        if mask[i]:
            last_val = series[i]
        elif not np.isnan(last_val):
            filled[i] = last_val
            filled_mask[i] = True
    return filled, filled_mask


def _extract_observed_blocks(series: np.ndarray, mask: np.ndarray, min_length: int = 12) -> list[tuple[int, np.ndarray]]:
    """Extract contiguous observed blocks of at least min_length months."""
    blocks = []
    start = None
    for i in range(len(mask)):
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                block = series[start:i]
                if len(block) >= min_length:
                    blocks.append((start, block))
                start = None
    if start is not None:
        block = series[start:]
        if len(block) >= min_length:
            blocks.append((start, block))
    return blocks


def _pure_python_pelt(signal: np.ndarray, pen: float, min_size: int = 3) -> list[int]:
    """Pure-Python implementation of PELT for 1D changepoint detection.
    Used when ruptures C-extensions are unavailable or blocked by OS policies.
    """
    n = len(signal)
    if n < 2 * min_size:
        return []

    prefix_sum = np.zeros(n + 1)
    prefix_sq = np.zeros(n + 1)
    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + signal[i]
        prefix_sq[i + 1] = prefix_sq[i] + signal[i] ** 2

    def cost(s: int, t: int) -> float:
        length = t - s
        if length <= 0:
            return 0.0
        sm = prefix_sum[t] - prefix_sum[s]
        sq = prefix_sq[t] - prefix_sq[s]
        return max(0.0, sq - (sm ** 2) / length)

    F = np.zeros(n + 1)
    F[0] = -pen
    cp_pointers = [0] * (n + 1)
    R = [0]

    for t in range(min_size, n + 1):
        candidates = [s for s in R if t - s >= min_size]
        if not candidates:
            continue

        best_cost = float("inf")
        best_s = 0
        for s in candidates:
            c = F[s] + cost(s, t) + pen
            if c < best_cost:
                best_cost = c
                best_s = s

        F[t] = best_cost
        cp_pointers[t] = best_s
        new_R = [t]
        for s in R:
            if (t - s < min_size) or (F[s] + cost(s, t) <= F[t]):
                new_R.append(s)
        R = new_R

    changepoints = []
    curr = n
    while curr > 0:
        prev = cp_pointers[curr]
        if prev > 0:
            changepoints.append(prev)
        curr = prev

    return sorted(changepoints)


def detect_pelt(
    series: np.ndarray,
    mask: np.ndarray,
    penalty: float | None = None,
    min_size: int = 3,
    beta: float = 2.0,
) -> list[int]:
    """Run PELT changepoint detection with rank/least-squares cost."""
    if len(series) != len(mask):
        raise ValueError("series and mask must have identical length")

    blocks = _extract_observed_blocks(series, mask, min_length=max(min_size * 2, 6))
    changepoints = []

    for start_idx, block in blocks:
        T = len(block)
        pen = penalty if penalty is not None else beta * np.log(T)
        signal = block.reshape(-1, 1).astype(np.float64)

        fitted_rpt = False
        if rpt is not None:
            try:
                algo = rpt.Pelt(model="rank", min_size=min_size, jump=1).fit(signal)
                bkps = algo.predict(pen=pen)
                for bp in bkps[:-1]:
                    changepoints.append(start_idx + bp)
                fitted_rpt = True
            except Exception:
                fitted_rpt = False

        if not fitted_rpt:
            # Fall back to pure Python PELT
            py_cps = _pure_python_pelt(block, pen=pen, min_size=min_size)
            for bp in py_cps:
                changepoints.append(start_idx + bp)

    return sorted(set(changepoints))


def detect_pelt_product(
    product_df: pd.DataFrame,
    aspects: list[str],
    beta: float = 2.0,
    min_size: int = 3,
    score_col: str = "norm_score",
) -> pd.DataFrame:
    """Run PELT across all aspects of a single product."""
    months = sorted(product_df["year_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months)}

    results = []
    for asp in aspects:
        asp_data = product_df[product_df["canonical_aspect"] == asp].sort_values("year_month")
        if len(asp_data) == 0:
            continue

        series = np.full(len(months), 0.0)
        mask = np.zeros(len(months), dtype=bool)

        for _, row in asp_data.iterrows():
            idx = month_to_idx.get(row["year_month"])
            if idx is not None:
                val = row.get(score_col, row.get("norm_score"))
                series[idx] = val if pd.notna(val) else 0.0
                mask[idx] = bool(row["aspect_mask"])

        cp_indices = detect_pelt(series, mask, beta=beta, min_size=min_size)

        for i, m in enumerate(months):
            results.append({
                "year_month": m,
                "canonical_aspect": asp,
                "is_changepoint": i in cp_indices,
            })

    return pd.DataFrame(results)


def pelt_fuse(detections_df: pd.DataFrame, tolerance: int = 1) -> pd.DataFrame:
    """Merge per-aspect PELT changepoints within +/- tolerance months into product events."""
    cps = detections_df[detections_df["is_changepoint"]].copy()
    if len(cps) == 0:
        return pd.DataFrame(columns=["event_month", "aspects_involved", "n_aspects"])

    cps = cps.sort_values("year_month").reset_index(drop=True)
    months = sorted(cps["year_month"].unique())

    clusters = []
    current_cluster = [months[0]]
    for m in months[1:]:
        prev = current_cluster[-1]
        diff = (m.year - prev.year) * 12 + (m.month - prev.month)
        if diff <= tolerance:
            current_cluster.append(m)
        else:
            clusters.append(current_cluster)
            current_cluster = [m]
    clusters.append(current_cluster)

    events = []
    for cluster_months in clusters:
        cluster_cps = cps[cps["year_month"].isin(cluster_months)]
        aspects = sorted(cluster_cps["canonical_aspect"].unique())
        events.append({
            "event_month": min(cluster_months),
            "aspects_involved": aspects,
            "n_aspects": len(aspects),
        })
    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# LSTM Autoencoder Architecture & Dataset
# ---------------------------------------------------------------------------
class SentimentWindowDataset(Dataset):
    """Sliding-window dataset for multivariate aspect time series.

    Input dimension: 18 features (9 aspect normalized scores + 9 observation mask channels).
    Target: 9 aspect scores (reconstructed over observed cells).
    """

    def __init__(
        self,
        product_dfs: dict[str, pd.DataFrame],
        aspects: list[str],
        seq_len: int = 6,
        stride: int = 1,
        exclude_warmup_targets: bool = True,
        score_col: str = "norm_score",
    ):
        self.seq_len = seq_len
        self.aspects = aspects
        self.n_aspects = len(aspects)
        self.windows = []

        for asin, df in product_dfs.items():
            months = sorted(df["year_month"].unique())
            if len(months) < seq_len:
                continue

            score_mat = np.zeros((len(months), self.n_aspects), dtype=np.float32)
            mask_mat = np.zeros((len(months), self.n_aspects), dtype=np.float32)
            warmup_vec = np.zeros(len(months), dtype=bool)

            month_to_idx = {m: i for i, m in enumerate(months)}
            asp_to_idx = {a: j for j, a in enumerate(aspects)}

            for _, row in df.iterrows():
                mi = month_to_idx.get(row["year_month"])
                ai = asp_to_idx.get(row["canonical_aspect"])
                if mi is None or ai is None:
                    continue
                score_val = row.get(score_col, row.get("norm_score"))
                if pd.notna(score_val) and row["aspect_mask"]:
                    score_mat[mi, ai] = float(score_val)
                    mask_mat[mi, ai] = 1.0
                if row.get("is_warmup", False):
                    warmup_vec[mi] = True

            for start in range(0, len(months) - seq_len + 1, stride):
                end = start + seq_len
                scores_window = score_mat[start:end]
                masks_window = mask_mat[start:end]

                # 18-dim input: scores (0 where unobserved) concatenated with binary observation masks
                input_tensor = np.concatenate([scores_window, masks_window], axis=1)
                target_tensor = scores_window.copy()
                target_mask = masks_window.copy()

                if exclude_warmup_targets:
                    for t in range(seq_len):
                        if warmup_vec[start + t]:
                            target_mask[t, :] = 0.0

                if target_mask.sum() == 0:
                    continue

                meta = {
                    "parent_asin": asin,
                    "window_start": months[start],
                    "window_end": months[end - 1],
                    "last_month": months[end - 1],
                    "n_observed": int(target_mask.sum()),
                }

                self.windows.append((
                    torch.tensor(input_tensor, dtype=torch.float32),
                    torch.tensor(target_tensor, dtype=torch.float32),
                    torch.tensor(target_mask, dtype=torch.float32),
                    meta,
                ))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        return self.windows[idx]


def collate_fn(batch):
    """Custom DataLoader collate function separating tensors from metadata dicts."""
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    metas = [b[3] for b in batch]
    return inputs, targets, masks, metas


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for multivariate temporal anomaly detection.

    Encoder: LSTM(18 -> 32) -> Linear(32 -> 8) -> ReLU
    Decoder: Linear(8 -> 32) -> LSTM(32 -> 32) -> Linear(32 -> 9)
    """

    def __init__(
        self,
        input_dim: int = 18,
        output_dim: int = 9,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder_dropout = nn.Dropout(dropout)
        self.encoder_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent_to_decoder = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.decoder_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        _, (h_n, _) = self.encoder_lstm(x)
        h = self.encoder_dropout(h_n.squeeze(0))
        latent = torch.relu(self.encoder_to_latent(h))

        decoder_input = torch.relu(self.latent_to_decoder(latent)).unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input)
        decoder_out = self.decoder_dropout(decoder_out)
        return self.output_layer(decoder_out)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE loss averaging per-aspect first so common aspects do not dominate."""
    squared_error = (pred - target) ** 2 * mask
    aspect_counts = mask.sum(dim=(0, 1))
    aspect_mse = squared_error.sum(dim=(0, 1)) / aspect_counts.clamp(min=1)
    active = (aspect_counts > 0).float()
    if active.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return (aspect_mse * active).sum() / active.sum()


def train_model(
    model: LSTMAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Train the LSTM Autoencoder with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for inputs, targets, masks, _ in train_loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = masked_mse_loss(preds, targets, masks)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.inference_mode():
            for inputs, targets, masks, _ in val_loader:
                inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
                preds = model(inputs)
                val_loss += masked_mse_loss(preds, targets, masks).item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def score_anomalies(
    model: LSTMAutoencoder,
    dataset: SentimentWindowDataset,
    threshold: Optional[float] = None,
    device: str = "cpu",
    batch_size: int = 128,
) -> pd.DataFrame:
    """Compute anomaly reconstruction error across sliding windows (max error over last 3 timesteps)."""
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_scores, all_metas = [], []

    with torch.inference_mode():
        for inputs, targets, masks, metas in loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
            preds = model(inputs)

            n_score_steps = min(3, preds.shape[1])
            step_errors = []
            for t in range(-n_score_steps, 0):
                t_pred = preds[:, t, :]
                t_target = targets[:, t, :]
                t_mask = masks[:, t, :]
                sq_err = (t_pred - t_target) ** 2 * t_mask
                n_obs = t_mask.sum(dim=1).clamp(min=1)
                step_errors.append(sq_err.sum(dim=1) / n_obs)

            stacked = torch.stack(step_errors, dim=1)
            sample_error = stacked.max(dim=1).values
            all_scores.extend(sample_error.cpu().numpy().tolist())
            all_metas.extend(metas)

    scores = np.array(all_scores)
    if threshold is None:
        threshold = float(np.percentile(scores, 95)) if len(scores) > 0 else 0.5

    results = []
    for i, meta in enumerate(all_metas):
        results.append({
            "parent_asin": meta["parent_asin"],
            "last_month": meta["last_month"],
            "recon_error": round(float(scores[i]), 6),
            "is_anomaly": bool(scores[i] > threshold),
            "threshold": round(float(threshold), 6),
        })
    return pd.DataFrame(results)


def detect_lstm_ae_product(
    model: LSTMAutoencoder,
    prod_df: pd.DataFrame,
    aspects: list[str],
    threshold: Optional[float] = None,
    seq_len: int = 6,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run LSTM-AE anomaly detector on a single product."""
    asin = prod_df["parent_asin"].iloc[0] if not prod_df.empty else "unknown"
    dataset = SentimentWindowDataset({asin: prod_df}, aspects, seq_len=seq_len, stride=1)
    if len(dataset) == 0:
        return pd.DataFrame(columns=["parent_asin", "last_month", "recon_error", "is_anomaly", "threshold"])
    return score_anomalies(model, dataset, threshold=threshold, device=device)
