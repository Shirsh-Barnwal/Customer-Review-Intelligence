"""Unit tests for CUSUM, PELT, and LSTM Autoencoder drift detection."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.drift_detection import (
    LSTMAutoencoder,
    SentimentWindowDataset,
    collate_fn,
    cusum_fuse,
    detect_cusum,
    detect_cusum_product,
    detect_pelt,
    detect_pelt_product,
    masked_mse_loss,
    pelt_fuse,
    score_anomalies,
)
from src.utils import CANONICAL_ASPECTS


def test_cusum_shift_detection():
    # Constant 0.0 baseline, then sustained negative drop of -2.0 at index 10
    T = 20
    series = np.zeros(T)
    series[10:] = -2.0
    mask = np.ones(T, dtype=bool)

    res = detect_cusum(series, mask, k=0.5, h=3.0, cooldown=2)
    assert len(res["alarm_indices"]) > 0
    assert res["alarm_indices"][0] >= 10
    assert "negative" in res["alarm_directions"]


def test_pelt_step_detection():
    # Flat 0.0, then abrupt step to 3.0 at index 12
    T = 25
    series = np.zeros(T)
    series[12:] = 3.0
    mask = np.ones(T, dtype=bool)

    cps = detect_pelt(series, mask, beta=1.5, min_size=3)
    assert len(cps) > 0
    assert any(abs(cp - 12) <= 1 for cp in cps)


def test_lstm_ae_dimensions_and_loss():
    batch_size = 4
    seq_len = 6
    model = LSTMAutoencoder(input_dim=18, output_dim=9, hidden_dim=32, latent_dim=8)

    dummy_input = torch.randn(batch_size, seq_len, 18)
    dummy_target = torch.randn(batch_size, seq_len, 9)
    dummy_mask = torch.ones(batch_size, seq_len, 9)

    out = model(dummy_input)
    assert out.shape == (batch_size, seq_len, 9)

    loss = masked_mse_loss(out, dummy_target, dummy_mask)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_cusum_fuse():
    df = pd.DataFrame([
        {"year_month": pd.Timestamp("2023-01-01"), "canonical_aspect": "battery", "is_changepoint": True, "direction": "negative"},
        {"year_month": pd.Timestamp("2023-02-01"), "canonical_aspect": "software", "is_changepoint": True, "direction": "negative"},
        {"year_month": pd.Timestamp("2023-08-01"), "canonical_aspect": "camera", "is_changepoint": True, "direction": "positive"},
    ])
    fused = cusum_fuse(df, tolerance=1)
    assert len(fused) == 2
    assert "battery" in fused.iloc[0]["aspects_involved"]
    assert "software" in fused.iloc[0]["aspects_involved"]
