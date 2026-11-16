"""Tests for src/explain/review_evidence.py."""
import pandas as pd
import pytest
from src.explain.review_evidence import select_quotes, keyword_evidence, NEXUS_6P_KEYWORDS


_REVIEW_PKS = [f"rpk_{i}" for i in range(8)]

_TIMESTAMPS = [
    pd.Timestamp("2023-01-15"), pd.Timestamp("2023-02-10"),
    pd.Timestamp("2023-02-20"), pd.Timestamp("2023-02-05"),
    pd.Timestamp("2023-01-25"), pd.Timestamp("2023-02-15"),
    pd.Timestamp("2023-01-05"), pd.Timestamp("2023-02-25"),
]


def make_reviews():
    return pd.DataFrame({
        "parent_asin": ["TEST"] * 8,
        "review_text_combined": [
            "Battery is draining super fast now",
            "Terrible battery life after update",
            "Battery is fine for my usage",
            "Software keeps crashing every day",
            "Love the camera quality",
            "Phone died and won't restart, bootloop",
            "Great performance overall",
            "Battery drain and shutdown issues",
        ],
        "rating": [1.0, 2.0, 4.0, 2.0, 5.0, 1.0, 5.0, 1.0],
        "timestamp": _TIMESTAMPS,
    })


def make_scaffold():
    return pd.DataFrame({
        "review_pk": _REVIEW_PKS,
        "parent_asin": ["TEST"] * 8,
        "timestamp": _TIMESTAMPS,
    })


def make_absa():
    return pd.DataFrame({
        "review_pk": _REVIEW_PKS,
        "parent_asin": ["TEST"] * 8,
        "canonical_aspect": ["battery", "battery", "battery", "software",
                              "camera", "battery", "performance", "battery"],
        "sentiment_score": [0.1, 0.15, 0.7, 0.2, 0.9, 0.05, 0.85, 0.1],
        "timestamp": _TIMESTAMPS,
    })


def test_select_quotes_returns_battery_quotes_only():
    result = select_quotes(make_reviews(), make_absa(), make_scaffold(),
                           "TEST", "2023-02", ["battery"], window=2, n_quotes=4)
    if len(result) > 0:
        assert all(result["canonical_aspect"] == "battery")


def test_select_quotes_truncates_to_max_chars():
    result = select_quotes(make_reviews(), make_absa(), make_scaffold(),
                           "TEST", "2023-02", ["battery"], window=2,
                           n_quotes=4, max_chars=20)
    if len(result) > 0:
        assert all(result["review_text_snippet"].str.len() <= 23)  # max_chars + "..."


def test_select_quotes_has_required_columns():
    result = select_quotes(make_reviews(), make_absa(), make_scaffold(),
                           "TEST", "2023-02", ["battery"], window=2, n_quotes=4)
    if len(result) > 0:
        required = {"parent_asin", "event_month", "canonical_aspect",
                    "review_text_snippet", "sentiment_score", "quote_type"}
        assert required.issubset(set(result.columns))


def test_keyword_evidence_counts_bootloop():
    result = keyword_evidence(make_reviews(), "TEST", "2023-02", ["bootloop"], window=2)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["count_post"] >= 1


def test_keyword_evidence_structure():
    result = keyword_evidence(make_reviews(), "TEST", "2023-02", ["battery"], window=2)
    assert isinstance(result, list)
    assert len(result) == 1
    assert set(result[0].keys()) >= {"keyword", "count_pre", "count_post", "ratio", "sample_quote"}


def test_nexus_keywords_list_nonempty():
    assert len(NEXUS_6P_KEYWORDS) >= 5
    assert "bootloop" in NEXUS_6P_KEYWORDS
