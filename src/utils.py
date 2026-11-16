"""Shared utilities, paths, constants, and JSON serialization helpers.

All filesystem paths are derived dynamically from the repository root:
    src/utils.py -> src/ -> repository root
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project Directory Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"                  # Raw Amazon review datasets (unprocessed)
DATA_PROCESSED = ROOT / "data" / "processed"      # Filtered & deduplicated reviews
DATA_TIMESERIES = ROOT / "data" / "timeseries"    # Aggregated monthly time series per product
DATA_LABELS = ROOT / "data" / "labels"            # Gold-labelled validation data
MODELS_DIR = ROOT / "models"                      # Saved model weights (PyTorch, scikit-learn)
RESULTS_DIR = ROOT / "results"                    # Evaluation figures, logs, JSON metrics
DASHBOARD_DATA = ROOT / "dashboard" / "react" / "public" / "data"  # Data consumed by React frontend

# ---------------------------------------------------------------------------
# Canonical Aspects & Sentiment Definitions
# ---------------------------------------------------------------------------
CANONICAL_ASPECTS = [
    "audio",
    "battery",
    "build",
    "camera",
    "charging",
    "connectivity",
    "display",
    "performance",
    "software",
]

# Aspect color palette used across figures and dashboard UI
ASPECT_COLORS = {
    "battery": "#3b82f6",       # Blue
    "camera": "#10b981",        # Emerald
    "display": "#8b5cf6",       # Purple
    "software": "#f59e0b",      # Amber
    "build": "#64748b",         # Slate
    "performance": "#ef4444",    # Red
    "connectivity": "#06b6d4",   # Cyan
    "audio": "#ec4899",         # Pink
    "charging": "#f97316",      # Orange
}

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_SCORE_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# ---------------------------------------------------------------------------
# Aspect Synonyms Dictionary
# ---------------------------------------------------------------------------
ASPECT_SYNONYMS: dict[str, list[str]] = {
    "battery": [
        "battery", "battery life", "battery drain", "power", "mah",
        "power bank", "dies fast", "lasts all day",
    ],
    "camera": [
        "camera", "photo", "video", "megapixel", "lens", "selfie",
        "portrait mode", "zoom", "night mode", "picture",
    ],
    "display": [
        "display", "screen", "amoled", "oled", "lcd", "resolution",
        "brightness", "120hz", "refresh rate", "touch screen",
    ],
    "software": [
        "android", "ios", "software", "update", "app", "operating system",
        "bloatware", "ui", "interface", "firmware", "bug", "glitch",
    ],
    "build": [
        "build", "build quality", "design", "waterproof", "water resistant",
        "dust", "durable", "weight", "glass", "plastic", "premium", "metal",
        "gorilla glass", "sturdy",
    ],
    "performance": [
        "processor", "ram", "memory", "storage", "speed", "snapdragon",
        "lag", "slow", "fast", "performance", "responsive", "freeze",
        "overheat", "heats up",
    ],
    "connectivity": [
        "5g", "4g", "lte", "wifi", "wi-fi", "bluetooth", "nfc", "signal",
        "reception", "network", "gps", "cellular", "hotspot",
    ],
    "audio": [
        "speaker", "audio", "sound", "headphone", "microphone", "earpiece",
        "call quality", "volume", "dolby", "stereo",
    ],
    "charging": [
        "charger", "charging", "fast charging", "wireless charging",
        "usb-c", "usb c", "cable", "watt", "quick charge",
    ],
}

# Reverse lookup dictionary: keyword -> canonical aspect
# Sorted longest first so multi-word phrases match before substrings
_KEYWORD_TO_ASPECT: dict[str, str] = {}
for _aspect, _keywords in ASPECT_SYNONYMS.items():
    for _kw in sorted(_keywords, key=len, reverse=True):
        _KEYWORD_TO_ASPECT[_kw.lower()] = _aspect


# ---------------------------------------------------------------------------
# Reproducibility & Encoders
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seed across python random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy and Pandas types."""

    def default(self, obj: Any) -> Any:
        if obj is pd.NaT or (hasattr(pd, "NA") and obj is pd.NA):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return round(float(obj), 6)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj.date())
        if isinstance(obj, pd.Period):
            return str(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)
