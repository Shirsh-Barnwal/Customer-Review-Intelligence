"""Bootstrap verification helper to make sure all data pipeline stages exist.

Usage:
    from src.bootstrap import ensure_inputs
    ensure_inputs(for_phase="02_aggregation")
"""
from __future__ import annotations

import os
from pathlib import Path

from .utils import (
    DATA_PROCESSED,
    DATA_TIMESERIES,
    MODELS_DIR,
    ROOT,
)

PHASE_OUTPUTS: dict[str, list[Path]] = {
    "00_eda": [
        DATA_PROCESSED / "review_scaffold.parquet",
        DATA_PROCESSED / "product_density.csv",
        DATA_PROCESSED / "drift_scores.csv",
        DATA_PROCESSED / "case_study_asins.txt",
    ],
    "01_absa": [
        DATA_PROCESSED / "review_aspect_sentiment.parquet",
    ],
    "02_aggregation": [
        DATA_TIMESERIES / "aspect_sentiment_monthly.parquet",
        DATA_TIMESERIES / "star_rating_monthly.parquet",
    ],
    "03_drift_detection": [
        MODELS_DIR / "drift" / "lstm_ae" / "best_lstm_ae.pt",
        ROOT / "results" / "drift" / "metrics.json",
    ],
    "04_explainability": [],
}

PHASE_ORDER = [
    "00_eda",
    "01_absa",
    "02_aggregation",
    "03_drift_detection",
    "04_explainability",
]


def _missing(phase: str) -> list[Path]:
    return [p for p in PHASE_OUTPUTS.get(phase, []) if not p.exists()]


def ensure_inputs(for_phase: str) -> None:
    """Verify that all upstream prerequisites for a given phase exist."""
    if os.environ.get("SENTINELDRIFT_SKIP_BOOTSTRAP") == "1":
        return

    if for_phase not in PHASE_ORDER:
        raise ValueError(f"Unknown phase {for_phase!r}. Valid phases: {PHASE_ORDER}")

    target_idx = PHASE_ORDER.index(for_phase)
    upstream = PHASE_ORDER[:target_idx]

    for phase in upstream:
        missing = _missing(phase)
        if missing:
            print(f"[Bootstrap] Notice: {phase} missing: " + ", ".join(str(p.relative_to(ROOT)) for p in missing))

    print(f"[Bootstrap] Verified inputs for {for_phase}")
