"""Tests for src/explain/aspect_delta.py."""
import numpy as np
import pandas as pd
import pytest
from src.explain.aspect_delta import (
    compute_event_delta, rank_aspects, compute_all_events,
    bootstrap_delta_ci, random_window_null,
)

ASPECTS = ["audio", "battery", "build", "camera", "charging",
           "connectivity", "display", "performance", "software"]


def make_monthly_df(product="TEST", n_months=24, aspects=None):
    """Synthetic monthly data: battery drops 0.3 at month 13."""
    if aspects is None:
        aspects = ["battery", "software"]
    rows = []
    months = pd.period_range("2022-01", periods=n_months, freq="M")
    for i, m in enumerate(months):
        for asp in aspects:
            score = (0.6 if i < 12 else 0.3) if asp == "battery" else 0.5
            rows.append({
                "parent_asin": product,
                "year_month": pd.Timestamp(str(m)),
                "canonical_aspect": asp,
                "raw_mean_sentiment": score,
                "norm_score": score,
                "mention_count": 10,
                "aspect_mask": True,
            })
    return pd.DataFrame(rows)


def test_compute_event_delta_known_battery_drop():
    df = make_monthly_df()
    result = compute_event_delta(df, "TEST", "2023-01", ["battery", "software"], window=3)
    bat = result[result.canonical_aspect == "battery"].iloc[0]
    sof = result[result.canonical_aspect == "software"].iloc[0]
    assert abs(bat["delta"] - (-0.3)) < 0.05, f"Expected delta ~-0.3, got {bat['delta']}"
    assert abs(sof["delta"]) < 0.05, f"Software should have ~0 delta, got {sof['delta']}"


def test_compute_event_delta_uses_raw_sentiment():
    df = make_monthly_df()
    result = compute_event_delta(
        df, "TEST", "2023-01", ["battery"], window=3, score_col="raw_mean_sentiment"
    )
    assert not result["pre_mean"].isna().all()


def test_compute_event_delta_min_obs_flag():
    """Sparse data (obs < min_obs) should flag confidence=low."""
    # n_months=4 gives 2022-01..04; event at 2022-04 with min_obs=4:
    # post window has only 1 month in-range (2022-04), so obs_post=1 < 4 → "low"
    df = make_monthly_df(n_months=4)
    result = compute_event_delta(df, "TEST", "2022-04", ["battery"], window=3, min_obs=4)
    assert (result["confidence"] == "low").any()


def test_compute_event_delta_partial_window_flag():
    df = make_monthly_df(n_months=24)
    # Event at month 2 — pre-window clips to start
    result = compute_event_delta(df, "TEST", "2022-02", ["battery"], window=3)
    assert (result["partial_window"] == True).any()


def test_rank_aspects_sorted_by_abs_delta():
    df = pd.DataFrame({
        "canonical_aspect": ["battery", "software", "camera"],
        "delta": [-0.3, 0.05, -0.1],
        "abs_delta": [0.3, 0.05, 0.1],
        "coverage": [1.0, 1.0, 1.0],
        "mention_count_post": [10, 10, 10],
    })
    result = rank_aspects(df)
    assert result.iloc[0]["canonical_aspect"] == "battery"
    assert result.iloc[1]["canonical_aspect"] == "camera"


def test_rank_aspects_direction_labels():
    df = pd.DataFrame({
        "canonical_aspect": ["battery", "software"],
        "delta": [-0.3, 0.1],
        "abs_delta": [0.3, 0.1],
        "coverage": [1.0, 1.0],
        "mention_count_post": [10, 10],
    })
    result = rank_aspects(df)
    bat = result[result.canonical_aspect == "battery"].iloc[0]
    sof = result[result.canonical_aspect == "software"].iloc[0]
    assert bat["direction"] == "deterioration"
    assert sof["direction"] == "improvement"


def test_rank_aspects_has_rank_column():
    df = pd.DataFrame({
        "canonical_aspect": ["battery"],
        "delta": [-0.3],
        "abs_delta": [0.3],
        "coverage": [1.0],
        "mention_count_post": [10],
    })
    result = rank_aspects(df)
    assert "rank" in result.columns
    assert result.iloc[0]["rank"] == 1


def test_random_window_null_shape():
    df = make_monthly_df(n_months=24)
    null_df = random_window_null(df, "TEST", ["battery", "software"], window=3, n_samples=50)
    assert "delta" in null_df.columns
    assert "canonical_aspect" in null_df.columns
    assert len(null_df) > 0


def test_bootstrap_delta_ci_structure():
    absa_df = pd.DataFrame({
        "parent_asin": ["TEST"] * 20,
        "canonical_aspect": ["battery"] * 20,
        "sentiment_score": [0.6] * 10 + [0.3] * 10,
        "timestamp": (
            [pd.Timestamp("2022-10-15")] * 10 +
            [pd.Timestamp("2023-02-15")] * 10
        ),
    })
    result = bootstrap_delta_ci(
        absa_df, "TEST", "2023-01", "battery", window=3, n_boot=200
    )
    assert set(result.keys()) >= {"delta", "ci_lower", "ci_upper", "n_reviews_pre", "n_reviews_post"}
    if result["ci_lower"] is not None and result["ci_upper"] is not None:
        assert result["ci_lower"] <= result["delta"] <= result["ci_upper"]
