"""Tests for src/explain/recon_decompose.py."""
import numpy as np
import pandas as pd
import pytest
import torch
from src.drift.lstm_ae import LSTMAutoencoder
from src.explain.recon_decompose import (
    decompose_recon_error, compare_delta_recon, decompose_all_events,
)

ASPECTS = ["audio", "battery", "build", "camera", "charging",
           "connectivity", "display", "performance", "software"]


def make_model(device="cpu"):
    model = LSTMAutoencoder(input_dim=18, output_dim=9, hidden_dim=32, latent_dim=8)
    return model.to(device).eval()


def make_monthly_df(product="TEST", n_months=12):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_months):
        month = pd.Period("2023-01", freq="M") + i
        for asp in ASPECTS:
            rows.append({
                "parent_asin": product,
                "year_month": pd.Timestamp(str(month)),
                "canonical_aspect": asp,
                "norm_score": float(rng.uniform(0.3, 0.7)),
                "raw_mean_sentiment": float(rng.uniform(0.3, 0.7)),
                "mention_count": 5,
                "aspect_mask": True,
            })
    return pd.DataFrame(rows)


def test_contributions_sum_to_100():
    model = make_model()
    df = make_monthly_df()
    result = decompose_recon_error(model, df, "TEST", "2023-06", ASPECTS, seq_len=6, device="cpu")
    assert set(result.columns) >= {"canonical_aspect", "recon_error", "mask", "contribution_pct"}
    observed = result[result["mask"] == True]
    assert abs(observed["contribution_pct"].sum() - 100.0) < 0.5


def test_unobserved_aspect_zero_contribution():
    model = make_model()
    df = make_monthly_df()
    # Remove "audio" from the last 3 months of the window (the timesteps
    # that could trigger the max-over-last-3 detection scoring).
    # Window: seq_len=6 ending at 2023-06, so last 3 = 2023-04, 2023-05, 2023-06.
    last3_periods = [pd.Period(f"2023-0{m}", "M") for m in [4, 5, 6]]
    df = df[~((df["canonical_aspect"] == "audio") &
              (df["year_month"].dt.to_period("M").isin(last3_periods)))]
    result = decompose_recon_error(model, df, "TEST", "2023-06", ASPECTS, seq_len=6, device="cpu")
    audio_row = result[result["canonical_aspect"] == "audio"]
    if len(audio_row) > 0:
        assert audio_row.iloc[0]["contribution_pct"] == pytest.approx(0.0, abs=0.01)


def test_output_has_all_aspects():
    model = make_model()
    df = make_monthly_df()
    result = decompose_recon_error(model, df, "TEST", "2023-06", ASPECTS, seq_len=6, device="cpu")
    assert set(result["canonical_aspect"].tolist()) == set(ASPECTS)


def test_compare_delta_recon_strong_agreement():
    deltas = pd.DataFrame({
        "canonical_aspect": ASPECTS,
        "abs_delta": [0.5, 0.4, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03],
        "delta": [-0.5, -0.4, -0.1, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03],
    })
    recon = pd.DataFrame({
        "canonical_aspect": ASPECTS,
        "recon_error": [0.5, 0.3, 0.2, 0.1, 0.08, 0.07, 0.05, 0.03, 0.01],
        "mask": [True] * 9,
    })
    result = compare_delta_recon(deltas, recon, top_k=3)
    assert set(result.keys()) >= {"top_k_delta", "top_k_recon", "overlap", "overlap_aspects",
                                   "agreement", "interpretation", "spearman_rho", "n_compared"}
    assert result["overlap"] >= 2
    assert result["agreement"] == "strong"
    assert result["n_compared"] == 9  # all aspects jointly observed
    assert result["spearman_rho"] is not None


def test_compare_delta_recon_no_agreement():
    deltas = pd.DataFrame({
        "canonical_aspect": ASPECTS,
        "abs_delta": [0.5, 0.4, 0.3, 0.1, 0.1, 0.05, 0.04, 0.03, 0.01],
        "delta": [-0.5, -0.4, -0.3, -0.1, -0.1, -0.05, -0.04, -0.03, -0.01],
    })
    recon = pd.DataFrame({
        "canonical_aspect": ASPECTS,
        "recon_error": [0.01, 0.02, 0.03, 0.4, 0.5, 0.3, 0.05, 0.04, 0.03],
        "mask": [True] * 9,
    })
    result = compare_delta_recon(deltas, recon, top_k=3)
    assert result["agreement"] in ("partial", "none")
