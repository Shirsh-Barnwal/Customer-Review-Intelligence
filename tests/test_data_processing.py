"""Unit tests for data processing, shrinkage, and normalization."""
import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    apply_shrinkage,
    flag_warmup,
    make_review_pk,
    normalize_mad,
)


def test_make_review_pk():
    pk1 = make_review_pk("user1", "ASIN123", "Great phone", "2023-01-01")
    pk2 = make_review_pk("user1", "ASIN123", "Great phone", "2023-01-01")
    pk3 = make_review_pk("user2", "ASIN123", "Great phone", "2023-01-01")
    assert pk1 == pk2
    assert pk1 != pk3
    assert len(pk1) == 16


def test_shrinkage_sparse_vs_dense():
    # n=1 should be shrunk toward prior, n>=3 should remain raw mean
    df = pd.DataFrame([
        {"parent_asin": "P1", "year_month": pd.Timestamp("2023-01-01"), "canonical_aspect": "battery", "mention_count": 1, "raw_mean_sentiment": -0.8, "aspect_mask": True, "review_count": 5, "is_warmup": False},
        {"parent_asin": "P1", "year_month": pd.Timestamp("2023-02-01"), "canonical_aspect": "battery", "mention_count": 10, "raw_mean_sentiment": 0.5, "aspect_mask": True, "review_count": 20, "is_warmup": False},
    ])
    shrunk = apply_shrinkage(df, k=5.0)
    assert pd.notna(shrunk.iloc[0]["shrunk_score"])
    # Row 1 (n=10) should keep its exact raw mean (0.5)
    assert abs(shrunk.iloc[1]["shrunk_score"] - 0.5) < 1e-5
    # Row 0 (n=1) should be pulled higher than -0.8 toward the lifetime prior
    assert shrunk.iloc[0]["shrunk_score"] > -0.8
