"""Explainability module for root-cause attribution and model interpretability.

Provides:
1. Aspect Delta Ranking: coarse-grained before/after sentiment change around detected events
2. Reconstruction Decomposition: per-aspect MSE contribution from the LSTM Autoencoder
3. Captum Integrated Gradients: fine-grained token-level attributions for PyABSA
4. Review Evidence Extraction: representative (median) and extreme review quotes
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn

from .utils import CANONICAL_ASPECTS, RESULTS_DIR

__all__ = [
    # Aspect Delta Ranking
    "compute_event_delta",
    "rank_aspects",
    "compute_all_events",
    "bootstrap_delta_ci",
    "random_window_null",
    # Reconstruction Decomposition
    "decompose_recon_error",
    "compare_delta_recon",
    "decompose_all_events",
    # Token Attributions (Captum)
    "compute_attributions",
    "render_heatmap",
    # Review Evidence
    "select_quotes",
    "keyword_evidence",
    "NEXUS_6P_KEYWORDS",
]

# Real-world Nexus 6P bootloop issue keywords
NEXUS_6P_KEYWORDS = [
    "bootloop", "bricked", "stuck", "restart", "won't turn on",
    "RMA", "warranty", "replacement", "overheat", "battery drain",
    "shutdown", "freeze", "unresponsive",
]


# ---------------------------------------------------------------------------
# 1. Aspect Delta Ranking
# ---------------------------------------------------------------------------
def compute_event_delta(
    monthly_df: pd.DataFrame,
    product: str,
    event_month: str | pd.Timestamp,
    aspects: list[str],
    window: int = 3,
    min_obs: int = 2,
    score_col: str = "raw_mean_sentiment",
    count_col: str = "mention_count",
) -> pd.DataFrame:
    """Compute pre vs post mean sentiment delta around a detected drift event."""
    event_month = pd.Timestamp(event_month)
    pdf = monthly_df[monthly_df["parent_asin"] == product].copy()

    if pdf.empty:
        return pd.DataFrame({
            "canonical_aspect": aspects,
            "pre_mean": np.nan,
            "post_mean": np.nan,
            "delta": np.nan,
            "abs_delta": np.nan,
            "coverage": 0.0,
            "confidence": "low",
            "mention_count_pre": 0,
            "mention_count_post": 0,
            "obs_pre": 0,
            "obs_post": 0,
            "partial_window": True,
        })

    pdf["year_month"] = pd.to_datetime(pdf["year_month"])
    t_min = pdf["year_month"].min()
    t_max = pdf["year_month"].max()

    pre_start = event_month - pd.DateOffset(months=window)
    pre_end = event_month
    post_start = event_month
    post_end = event_month + pd.DateOffset(months=window)

    partial = (pre_start < t_min) or (post_end > t_max + pd.DateOffset(months=1))
    results = []

    for asp in aspects:
        asp_data = pdf[pdf["canonical_aspect"] == asp]
        pre = asp_data[(asp_data["year_month"] >= pre_start) & (asp_data["year_month"] < pre_end) & asp_data["aspect_mask"]]
        post = asp_data[(asp_data["year_month"] >= post_start) & (asp_data["year_month"] < post_end) & asp_data["aspect_mask"]]

        obs_pre = len(pre)
        obs_post = len(post)
        pre_mean = float(pre[score_col].mean()) if obs_pre > 0 else np.nan
        post_mean = float(post[score_col].mean()) if obs_post > 0 else np.nan

        delta = (post_mean - pre_mean) if (obs_pre > 0 and obs_post > 0) else np.nan
        coverage = (obs_pre + obs_post) / (2 * window) if window > 0 else 0.0
        conf = "high" if (obs_pre >= min_obs and obs_post >= min_obs) else "low"

        results.append({
            "canonical_aspect": asp,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "delta": delta,
            "abs_delta": abs(delta) if pd.notna(delta) else np.nan,
            "coverage": coverage,
            "confidence": conf,
            "mention_count_pre": int(pre[count_col].sum()) if obs_pre > 0 else 0,
            "mention_count_post": int(post[count_col].sum()) if obs_post > 0 else 0,
            "obs_pre": obs_pre,
            "obs_post": obs_post,
            "partial_window": partial,
        })

    return pd.DataFrame(results)


def rank_aspects(deltas_df: pd.DataFrame) -> pd.DataFrame:
    """Sort aspects by absolute delta descending with direction tags."""
    df = deltas_df.copy()
    df["direction"] = np.select(
        [df["delta"].isna(), df["delta"] < 0, df["delta"] > 0],
        ["unknown", "deterioration", "improvement"],
        default="neutral",
    )
    df = df.sort_values(
        ["abs_delta", "coverage", "mention_count_post"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def compute_all_events(
    monthly_df: pd.DataFrame,
    events_df: pd.DataFrame,
    aspects: list[str],
    **kwargs,
) -> pd.DataFrame:
    """Batch compute and rank deltas for all detected events."""
    all_deltas = []
    for _, event in events_df.iterrows():
        delta_df = compute_event_delta(
            monthly_df=monthly_df,
            product=event["parent_asin"],
            event_month=event["event_month"],
            aspects=aspects,
            **kwargs,
        )
        delta_df = rank_aspects(delta_df)
        delta_df["parent_asin"] = event["parent_asin"]
        delta_df["event_month"] = event["event_month"]
        delta_df["detector"] = event.get("detector", "unknown")
        all_deltas.append(delta_df)

    return pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.DataFrame()


def bootstrap_delta_ci(
    absa_df: pd.DataFrame,
    product: str,
    event_month: str | pd.Timestamp,
    aspect: str,
    window: int = 3,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Resample review scores within pre and post windows to compute bootstrap confidence interval."""
    event_month = pd.Timestamp(event_month)
    pre_start = event_month - pd.DateOffset(months=window)
    post_end = event_month + pd.DateOffset(months=window)

    mask = (absa_df["parent_asin"] == product) & (absa_df["canonical_aspect"] == aspect)
    asp_reviews = absa_df[mask].copy()
    asp_reviews["timestamp"] = pd.to_datetime(asp_reviews["timestamp"])

    pre = asp_reviews[(asp_reviews["timestamp"] >= pre_start) & (asp_reviews["timestamp"] < event_month)]["sentiment_score"].values
    post = asp_reviews[(asp_reviews["timestamp"] >= event_month) & (asp_reviews["timestamp"] < post_end)]["sentiment_score"].values

    if len(pre) == 0 or len(post) == 0:
        return {"delta": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_reviews_pre": len(pre), "n_reviews_post": len(post)}

    rng = np.random.default_rng(42)
    boot_deltas = np.empty(n_boot)
    for b in range(n_boot):
        pre_sample = rng.choice(pre, size=len(pre), replace=True)
        post_sample = rng.choice(post, size=len(post), replace=True)
        boot_deltas[b] = post_sample.mean() - pre_sample.mean()

    alpha = 1 - ci
    return {
        "delta": float(post.mean() - pre.mean()),
        "ci_lower": float(np.percentile(boot_deltas, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_deltas, 100 * (1 - alpha / 2))),
        "n_reviews_pre": len(pre),
        "n_reviews_post": len(post),
    }


def random_window_null(
    monthly_df: pd.DataFrame,
    product: str,
    aspects: list[str],
    window: int = 3,
    n_samples: int = 500,
    score_col: str = "raw_mean_sentiment",
) -> pd.DataFrame:
    """Sample random pseudo-event windows to establish an empirical null distribution."""
    pdf = monthly_df[monthly_df["parent_asin"] == product].copy()
    pdf["year_month"] = pd.to_datetime(pdf["year_month"])
    months = sorted(pdf["year_month"].unique())

    if len(months) < 2 * window + 1:
        return pd.DataFrame(columns=["sample_idx", "canonical_aspect", "delta"])

    valid_start = months[0] + pd.DateOffset(months=window)
    valid_end = months[-1] - pd.DateOffset(months=window)
    valid_months = [m for m in months if valid_start <= m <= valid_end]
    if not valid_months:
        return pd.DataFrame(columns=["sample_idx", "canonical_aspect", "delta"])

    rng = np.random.default_rng(42)
    chosen = rng.choice(len(valid_months), size=n_samples, replace=True)
    aspect_map = {asp: pdf[(pdf["canonical_aspect"] == asp) & pdf["aspect_mask"].fillna(False)] for asp in aspects}

    results = []
    for i, idx in enumerate(chosen):
        split = valid_months[idx]
        for asp in aspects:
            adata = aspect_map[asp]
            pre = adata[(adata["year_month"] >= split - pd.DateOffset(months=window)) & (adata["year_month"] < split)][score_col]
            post = adata[(adata["year_month"] >= split) & (adata["year_month"] < split + pd.DateOffset(months=window))][score_col]
            delta = float(post.mean() - pre.mean()) if (len(pre) > 0 and len(post) > 0) else np.nan
            results.append({"sample_idx": i, "canonical_aspect": asp, "delta": delta})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2. Reconstruction Decomposition (LSTM-AE)
# ---------------------------------------------------------------------------
def decompose_recon_error(
    model: nn.Module,
    monthly_df: pd.DataFrame,
    product: str,
    event_month: str | pd.Timestamp,
    aspects: list[str],
    seq_len: int = 6,
    device: str | None = None,
    score_col: str = "norm_score",
) -> pd.DataFrame:
    """Decompose LSTM-AE anomaly score at an event into per-aspect percentage contributions."""
    event_month = pd.Timestamp(event_month)
    n_aspects = len(aspects)
    pdf = monthly_df[monthly_df["parent_asin"] == product].copy()
    pdf["year_month"] = pd.to_datetime(pdf["year_month"])
    months = sorted(pdf["year_month"].unique())

    if len(months) == 0:
        return pd.DataFrame({"canonical_aspect": aspects, "recon_error": 0.0, "mask": False, "contribution_pct": 0.0})

    diffs = [abs((m - event_month).days) for m in months]
    event_idx = int(np.argmin(diffs))

    window_end = event_idx + 1
    window_start = max(0, window_end - seq_len)
    actual_len = window_end - window_start
    if actual_len < 1:
        return pd.DataFrame({"canonical_aspect": aspects, "recon_error": 0.0, "mask": False, "contribution_pct": 0.0})

    asp_to_idx = {a: j for j, a in enumerate(aspects)}
    score_mat = np.zeros((actual_len, n_aspects), dtype=np.float32)
    mask_mat = np.zeros((actual_len, n_aspects), dtype=np.float32)

    for t_offset, t_abs in enumerate(range(window_start, window_end)):
        m = months[t_abs]
        mdata = pdf[pdf["year_month"] == m]
        for _, row in mdata.iterrows():
            ai = asp_to_idx.get(row["canonical_aspect"])
            if ai is not None and row["aspect_mask"] and pd.notna(row.get(score_col)):
                score_mat[t_offset, ai] = float(row[score_col])
                mask_mat[t_offset, ai] = 1.0

    if actual_len < seq_len:
        pad_len = seq_len - actual_len
        score_mat = np.pad(score_mat, ((pad_len, 0), (0, 0)), mode="constant")
        mask_mat = np.pad(mask_mat, ((pad_len, 0), (0, 0)), mode="constant")

    input_tensor = np.concatenate([score_mat, mask_mat], axis=1)
    x = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)

    model_device = torch.device(device) if device is not None else next(model.parameters()).device
    x = x.to(model_device)
    model.eval()
    with torch.inference_mode():
        x_hat = model(x)

    n_score_steps = min(3, score_mat.shape[0])
    best_t = -1
    best_total = -1.0
    for t in range(-n_score_steps, 0):
        t_scores = x[0, t, :n_aspects].cpu().numpy()
        t_recon = x_hat[0, t, :].cpu().numpy()
        t_mask = mask_mat[t, :]
        t_err = t_mask * (t_scores - t_recon) ** 2
        t_total = t_err.sum() / max(t_mask.sum(), 1.0)
        if t_total > best_total:
            best_total = t_total
            best_t = t

    trigger_scores = x[0, best_t, :n_aspects].cpu().numpy()
    trigger_recon = x_hat[0, best_t, :].cpu().numpy()
    trigger_mask = mask_mat[best_t, :]
    per_aspect_err = trigger_mask * (trigger_scores - trigger_recon) ** 2
    total_err = per_aspect_err.sum()
    contrib = (per_aspect_err / total_err * 100.0) if total_err > 0 else np.zeros(n_aspects)

    return pd.DataFrame({
        "canonical_aspect": aspects,
        "recon_error": per_aspect_err,
        "mask": trigger_mask.astype(bool),
        "contribution_pct": contrib,
    })


def compare_delta_recon(
    deltas_df: pd.DataFrame,
    recon_df: pd.DataFrame,
    top_k: int = 3,
) -> dict:
    """Compare aspect delta ranking with reconstruction error ranking."""
    delta_sorted = deltas_df.dropna(subset=["abs_delta"]).sort_values("abs_delta", ascending=False)
    top_delta = delta_sorted["canonical_aspect"].head(top_k).tolist()

    recon_sort_col = "recon_error" if "recon_error" in recon_df.columns else "contribution_pct"
    mask_col = recon_df["mask"] if "mask" in recon_df.columns else pd.Series([True] * len(recon_df))
    recon_observed = recon_df[mask_col].sort_values(recon_sort_col, ascending=False)
    top_recon = recon_observed["canonical_aspect"].head(top_k).tolist()

    overlap_set = set(top_delta) & set(top_recon)
    overlap = len(overlap_set)

    joint = delta_sorted.merge(
        recon_observed[["canonical_aspect", recon_sort_col]],
        on="canonical_aspect",
        how="inner",
    )
    n_compared = len(joint)
    if n_compared >= 4:
        rho, p_val = spearmanr(joint["abs_delta"], joint[recon_sort_col])
        spearman_rho = round(float(rho), 3)
        spearman_p = round(float(p_val), 4)
    else:
        spearman_rho = None
        spearman_p = None

    max_possible = min(len(top_delta), len(top_recon))
    if max_possible == 0:
        agreement = "none"
        interpretation = "No comparison possible: insufficient observed aspects for ranking."
    elif spearman_rho is not None:
        if spearman_rho >= 0.5 and (overlap / max_possible >= 0.66 if max_possible > 0 else False):
            agreement = "strong"
            interpretation = f"Strong agreement (Spearman rho={spearman_rho}, top-{top_k} overlap={overlap}/{max_possible})"
        elif spearman_rho >= 0.3 or overlap > 0:
            agreement = "partial"
            interpretation = f"Partial agreement (Spearman rho={spearman_rho}, top-{top_k} overlap={overlap}/{max_possible})"
        else:
            agreement = "none"
            interpretation = f"No agreement (Spearman rho={spearman_rho}, top-{top_k} overlap={overlap}/{max_possible})"
    else:
        ratio = (overlap / max_possible) if max_possible > 0 else 0.0
        if ratio >= 0.66:
            agreement = "strong"
        elif ratio > 0:
            agreement = "partial"
        else:
            agreement = "none"
        interpretation = f"Top-{top_k} overlap: {overlap}/{max_possible}"

    return {
        "top_k_delta": top_delta,
        "top_k_recon": top_recon,
        "overlap": overlap,
        "overlap_aspects": sorted(overlap_set),
        "agreement": agreement,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "n_compared": n_compared,
        "interpretation": interpretation,
        "top_k_overlap": overlap,
        "top_k_overlap_pct": round((overlap / max_possible * 100.0) if max_possible > 0 else 0.0, 1),
        "delta_top_aspects": top_delta,
        "recon_top_aspects": top_recon,
    }


def decompose_all_events(
    model: nn.Module,
    events_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    aspects: list[str],
    seq_len: int = 6,
    device: str | None = None,
) -> pd.DataFrame:
    """Batch decompose reconstruction error across all detected events."""
    all_recons = []
    for _, event in events_df.iterrows():
        rdf = decompose_recon_error(
            model=model,
            monthly_df=monthly_df,
            product=event["parent_asin"],
            event_month=event["event_month"],
            aspects=aspects,
            seq_len=seq_len,
            device=device,
        )
        rdf["parent_asin"] = event["parent_asin"]
        rdf["event_month"] = event["event_month"]
        rdf["detector"] = event.get("detector", "unknown")
        all_recons.append(rdf)

    return pd.concat(all_recons, ignore_index=True) if all_recons else pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. Token-Level Attribution via Captum Integrated Gradients
# ---------------------------------------------------------------------------
def compute_attributions(
    text: str,
    aspect: str,
    target_label: int | None = None,
    checkpoint: str = "english",
    n_steps: int = 20,
) -> dict:
    """Compute token-level Layer Integrated Gradients for PyABSA on a customer review."""
    from .absa_models import _load_classifier, inject_aspect_tags

    clf = _load_classifier(checkpoint)
    if clf is None:
        raise RuntimeError("PyABSA classifier is required for Captum Integrated Gradients.")

    match = re.search(rf"\b{re.escape(aspect)}\b", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Aspect '{aspect}' was not found in review text.")
    surface = match.group(0)

    tagged = text[:match.start()] + f"[B-ASP]{surface}[E-ASP]" + text[match.end():]
    clf.predict(tagged, print_result=False)

    dl = clf.infer_dataloader
    sample = list(iter(dl))[-1]
    inputs_cols = [c for c in clf.config.inputs_cols if c != "polarity"]
    device = clf.config.device
    inputs = {col: sample[col].to(device) for col in inputs_cols}
    text_indices = inputs["text_indices"]

    inner = clf.model.models[0]
    embedding_layer = inner.bert4global.get_input_embeddings()
    inner.eval()

    with torch.no_grad():
        logits_pred = inner(inputs)["logits"]
    predicted = int(logits_pred.argmax(dim=-1).item())
    target = int(target_label) if target_label is not None else predicted

    def forward_on_ids(ids: torch.Tensor) -> torch.Tensor:
        inp_loc = dict(inputs)
        inp_loc["text_indices"] = ids
        return inner(inp_loc)["logits"]

    from captum.attr import LayerIntegratedGradients
    lig = LayerIntegratedGradients(forward_on_ids, embedding_layer)
    attributions = lig.attribute(
        inputs=text_indices,
        baselines=torch.zeros_like(text_indices),
        target=target,
        n_steps=n_steps,
        internal_batch_size=4,
    )

    token_attrs = attributions.sum(dim=-1).squeeze(0)
    max_abs = float(token_attrs.abs().max().item()) or 1.0
    scores_full = (token_attrs / max_abs).detach().cpu().tolist()
    token_ids = text_indices.squeeze(0).tolist()

    tokenizer = getattr(clf, "tokenizer", None)
    hf_tok = getattr(tokenizer, "tokenizer", tokenizer)
    tokens_full = hf_tok.convert_ids_to_tokens(token_ids)

    pad_id = getattr(hf_tok, "pad_token_id", None)
    keep = [tid != pad_id for tid in token_ids] if pad_id is not None else [t not in {"[PAD]", "<pad>"} for t in tokens_full]
    tokens = [t for t, k in zip(tokens_full, keep) if k]
    scores = [s for s, k in zip(scores_full, keep) if k]

    label_names = ("negative", "neutral", "positive")
    return {
        "tokens": tokens,
        "scores": scores,
        "predicted_label": label_names[predicted],
        "confidence": float(torch.softmax(logits_pred.squeeze(0), dim=-1)[predicted].item()),
        "target_label": target,
    }


def render_heatmap(attributions: dict, save_to: str | Path | None = None) -> None:
    """Render and optionally save an attribution heatmap for review tokens."""
    import matplotlib.pyplot as plt

    tokens = attributions["tokens"]
    scores = np.array(attributions["scores"])
    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.4), 2))

    im = ax.imshow(scores.reshape(1, -1), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.3, label="Attribution weight")
    plt.title(f"Predicted: {attributions['predicted_label']} (conf: {attributions['confidence']:.2f})")
    plt.tight_layout()

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 4. Review Evidence Selection
# ---------------------------------------------------------------------------
def _dedup_quotes(quotes_df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Deduplicate near-identical quotes via TF-IDF cosine similarity."""
    texts = quotes_df["review_text_snippet"].tolist()
    if len(texts) < 2:
        return quotes_df

    try:
        vec = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf = vec.fit_transform(texts)
        sim = cosine_similarity(tfidf)
        keep = []
        for i in range(len(texts)):
            if not any(sim[i, j] > threshold for j in keep):
                keep.append(i)
        return quotes_df.iloc[keep].reset_index(drop=True)
    except Exception:
        return quotes_df


def select_quotes(
    reviews_df: pd.DataFrame,
    absa_df: pd.DataFrame,
    scaffold_df: pd.DataFrame,
    product: str,
    event_month: str | pd.Timestamp,
    top_aspects: list[str],
    window: int = 3,
    n_quotes: int = 6,
    max_chars: int = 400,
    dedup_threshold: float = 0.9,
    aspect_deltas: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Select representative (median sentiment) and extreme review quotes from post-event windows."""
    event_month = pd.Timestamp(event_month)
    post_start = event_month
    post_end = event_month + pd.DateOffset(months=window)

    scaffold_sub = scaffold_df[scaffold_df["parent_asin"] == product].drop(columns=["rating"], errors="ignore")
    reviews_sub = reviews_df[reviews_df["parent_asin"] == product].copy()

    candidates = ["review_text_combined", "review_text", "text", "body", "content"]
    text_col = next((c for c in candidates if c in reviews_sub.columns), None)
    if text_col is None:
        return pd.DataFrame(columns=[
            "parent_asin", "event_month", "canonical_aspect",
            "review_text_snippet", "sentiment_score", "quote_type",
            "review_month", "stars",
        ])
    if "review_pk" in reviews_sub.columns:
        joined = scaffold_sub.merge(reviews_sub[["review_pk", text_col, "rating"]], on="review_pk", how="inner")
    else:
        joined = scaffold_sub.merge(reviews_sub[["timestamp", text_col, "rating"]], on="timestamp", how="inner")

    absa_sub = absa_df[absa_df["parent_asin"] == product].copy()
    absa_sub["timestamp"] = pd.to_datetime(absa_sub["timestamp"])
    joined["timestamp"] = pd.to_datetime(joined["timestamp"])

    review_aspects = joined.merge(absa_sub[["review_pk", "canonical_aspect", "sentiment_score"]], on="review_pk", how="inner")
    review_aspects = review_aspects[(review_aspects["timestamp"] >= post_start) & (review_aspects["timestamp"] < post_end)]

    if review_aspects.empty:
        return pd.DataFrame(columns=[
            "parent_asin", "event_month", "canonical_aspect",
            "review_text_snippet", "sentiment_score", "quote_type",
            "review_month", "stars",
        ])

    review_aspects["review_month"] = review_aspects["timestamp"].dt.to_period("M").astype(str)
    all_quotes = []
    quotes_per_aspect = max(2, n_quotes // max(len(top_aspects), 1))

    for asp in top_aspects:
        asp_reviews = review_aspects[review_aspects["canonical_aspect"] == asp].drop_duplicates(subset=[text_col])
        if asp_reviews.empty:
            continue

        median_sent = asp_reviews["sentiment_score"].median()
        asp_reviews["_dist_to_median"] = (asp_reviews["sentiment_score"] - median_sent).abs()
        rep = asp_reviews.nsmallest(1, "_dist_to_median")

        extreme_cands = asp_reviews[~asp_reviews.index.isin(rep.index)]
        if extreme_cands.empty:
            extreme = pd.DataFrame()
        elif aspect_deltas and aspect_deltas.get(asp, 0) > 0:
            extreme = extreme_cands.nlargest(1, "sentiment_score")
        else:
            extreme = extreme_cands.nsmallest(1, "sentiment_score")

        added = 0
        for qtype, subset in [("representative", rep), ("extreme", extreme)]:
            if subset.empty:
                continue
            for _, row in subset.iterrows():
                txt = str(row[text_col])
                if len(txt) > max_chars:
                    txt = txt[:max_chars].rsplit(" ", 1)[0] + "..."
                stars = int(row["rating"]) if pd.notna(row["rating"]) else 0
                all_quotes.append({
                    "parent_asin": product,
                    "event_month": str(event_month.date()),
                    "canonical_aspect": asp,
                    "review_text_snippet": txt,
                    "sentiment_score": float(row["sentiment_score"]),
                    "quote_type": qtype,
                    "review_month": row["review_month"],
                    "stars": stars,
                })
                added += 1
            if added >= quotes_per_aspect:
                break

        if len(all_quotes) >= n_quotes:
            break

    result = pd.DataFrame(all_quotes)
    if len(result) > 1:
        result = _dedup_quotes(result, threshold=dedup_threshold)
    return result.head(n_quotes).reset_index(drop=True)


def keyword_evidence(
    reviews_df: pd.DataFrame,
    product: str,
    event_month: str | pd.Timestamp | None = None,
    keywords: list[str] | None = None,
    window: int = 3,
    max_quotes: int = 5,
) -> list[dict]:
    """Count keyword hits in pre vs post window, or find direct review evidence."""
    kws = keywords if keywords is not None else NEXUS_6P_KEYWORDS
    prod_reviews = reviews_df[reviews_df["parent_asin"] == product].copy()
    candidates = ["review_text_combined", "review_text", "text", "body", "content"]
    text_col = next((c for c in candidates if c in prod_reviews.columns), None)
    if text_col is None:
        return []

    if event_month is not None:
        event_month = pd.Timestamp(event_month)
        pre_start = event_month - pd.DateOffset(months=window)
        post_end = event_month + pd.DateOffset(months=window)

        pre = prod_reviews[
            (prod_reviews["timestamp"] >= pre_start)
            & (prod_reviews["timestamp"] < event_month)
        ]
        post = prod_reviews[
            (prod_reviews["timestamp"] >= event_month)
            & (prod_reviews["timestamp"] < post_end)
        ]

        results = []
        for kw in kws:
            kw_lower = kw.lower()
            pre_hits = pre[pre[text_col].str.contains(kw_lower, case=False, regex=False, na=False)]
            post_hits = post[post[text_col].str.contains(kw_lower, case=False, regex=False, na=False)]

            count_pre = len(pre_hits)
            count_post = len(post_hits)
            ratio = count_post / max(count_pre, 1)

            sample_quote = ""
            if len(post_hits) > 0:
                text = str(post_hits.iloc[0][text_col])
                sample_quote = text[:300].rsplit(" ", 1)[0] + "..." if len(text) > 300 else text

            results.append({
                "keyword": kw,
                "count_pre": count_pre,
                "count_post": count_post,
                "ratio": round(ratio, 2),
                "sample_quote": sample_quote,
            })
        return results

    matches = []
    for _, row in prod_reviews.iterrows():
        t = str(row[text_col]).lower()
        found_kw = [k for k in kws if re.search(r"\b" + re.escape(k) + r"\b", t)]
        if found_kw:
            matches.append({
                "parent_asin": product,
                "keywords": found_kw,
                "review_text": str(row[text_col]),
                "stars": int(row.get("rating", 0)),
                "date": str(pd.to_datetime(row["timestamp"]).date()) if pd.notna(row.get("timestamp")) else "",
            })
        if len(matches) >= max_quotes:
            break

    return matches
