"""Data preprocessing, scaffolding, and temporal aggregation pipeline.

Combines raw review filtering, primary key generation, and monthly temporal
aggregation with Empirical Bayes shrinkage and outlier-robust Median/MAD normalization.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

from .utils import CANONICAL_ASPECTS

__all__ = [
    "make_review_pk",
    "clean_reviews",
    "build_monthly_aspect_grid",
    "flag_warmup",
    "apply_shrinkage",
    "normalize_mad",
    "normalize_mad_reference",
    "build_star_rating_baseline",
]


# ---------------------------------------------------------------------------
# Review Primary Key & Cleaning Helpers
# ---------------------------------------------------------------------------
def make_review_pk(
    user_id: str,
    parent_asin: str,
    review_text: str,
    timestamp: Any,
) -> str:
    """Generate a deterministic 16-character SHA-256 hash as review primary key."""
    raw = f"{user_id}|{parent_asin}|{review_text}|{timestamp}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and deduplicate review dataframe.

    Ensures timestamp parsing, fills missing text, creates review_pk,
    and drops identical duplicates.
    """
    cleaned = df.copy()
    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"])

    text_col = "review_text_combined" if "review_text_combined" in cleaned.columns else "text"
    if text_col in cleaned.columns:
        cleaned[text_col] = cleaned[text_col].fillna("").astype(str)

    # Generate primary keys if missing
    if "review_pk" not in cleaned.columns and text_col in cleaned.columns:
        cleaned["review_pk"] = [
            make_review_pk(
                str(row.get("user_id", "")),
                str(row.get("parent_asin", "")),
                str(row.get(text_col, "")),
                str(row.get("timestamp", "")),
            )
            for _, row in cleaned.iterrows()
        ]

    # Deduplicate on review_pk
    if "review_pk" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["review_pk"]).reset_index(drop=True)

    return cleaned


# ---------------------------------------------------------------------------
# Temporal Aggregation Helpers
# ---------------------------------------------------------------------------
def _to_month_start(ts: pd.Series) -> pd.Series:
    """Convert dates to the first day of the calendar month."""
    return pd.to_datetime(ts).dt.to_period("M").dt.to_timestamp()


def _build_product_month_grid(scaffold: pd.DataFrame) -> pd.DataFrame:
    """Build a complete, contiguous calendar-month grid covering each product's active lifespan."""
    bounds = scaffold.groupby("parent_asin", as_index=False).agg(
        first_month=("year_month", "min"),
        last_month=("year_month", "max"),
    )
    rows = []
    for _, row in bounds.iterrows():
        months = pd.period_range(
            start=row["first_month"], end=row["last_month"], freq="M"
        ).to_timestamp()
        for m in months:
            rows.append({"parent_asin": row["parent_asin"], "year_month": m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monthly Aspect Grid Construction
# ---------------------------------------------------------------------------
def build_monthly_aspect_grid(
    aspect_df: pd.DataFrame,
    scaffold_df: pd.DataFrame,
    aspects: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build the product x month x aspect longitudinal grid with raw monthly means.

    Parameters
    ----------
    aspect_df : DataFrame with aspect-level sentiment predictions:
        [parent_asin, timestamp, canonical_aspect, sentiment_score].
    scaffold_df : DataFrame with review scaffold:
        [review_pk, parent_asin, timestamp, rating, verified_purchase].
    aspects : Optional list of canonical aspects. Defaults to CANONICAL_ASPECTS.

    Returns
    -------
    Long-format DataFrame with exactly one row per (parent_asin, year_month, aspect).
    """
    aspect = aspect_df.copy()
    aspect["year_month"] = _to_month_start(aspect["timestamp"])

    scaffold = scaffold_df.copy()
    scaffold["year_month"] = _to_month_start(scaffold["timestamp"])

    aspect_list = aspects or CANONICAL_ASPECTS

    # Full product-month grid ensures complete chronological continuity
    product_month_grid = _build_product_month_grid(scaffold)

    # Cross join with all aspects so unmentioned aspects have explicit rows
    aspect_levels = pd.DataFrame({"canonical_aspect": sorted(aspect_list)})
    full_grid = product_month_grid.merge(aspect_levels, how="cross")

    # Aggregate aspect predictions
    aspect_monthly = (
        aspect
        .groupby(["parent_asin", "year_month", "canonical_aspect"], as_index=False)
        .agg(
            raw_mean_sentiment=("sentiment_score", "mean"),
            mention_count=("sentiment_score", "size"),
        )
    )
    aspect_monthly["mention_count"] = aspect_monthly["mention_count"].astype(int)

    # Aggregate scaffold level review counts and verified purchase ratio
    scaffold_monthly = (
        scaffold
        .groupby(["parent_asin", "year_month"], as_index=False)
        .agg(
            review_count=("review_pk", "nunique"),
            verified_ratio=("verified_purchase", "mean"),
        )
    )

    # Join full grid with aggregated aspect and scaffold stats
    grid = (
        full_grid
        .merge(aspect_monthly, on=["parent_asin", "year_month", "canonical_aspect"], how="left")
        .merge(scaffold_monthly, on=["parent_asin", "year_month"], how="left")
        .sort_values(["parent_asin", "year_month", "canonical_aspect"])
        .reset_index(drop=True)
    )

    grid["mention_count"] = grid["mention_count"].fillna(0).astype(int)
    grid["review_count"] = grid["review_count"].fillna(0).astype(int)
    grid["log_review_count"] = np.log1p(grid["review_count"].astype(float))
    grid["verified_ratio"] = grid["verified_ratio"].fillna(0.0)
    grid["aspect_mask"] = grid["mention_count"] > 0

    return grid


def flag_warmup(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Identify and flag the very first month with actual reviews per product as warm-up.

    Warmup months have unstable review volumes and should not bias the normalization baseline.
    """
    grid = grid_df.copy()
    months_with_reviews = grid[grid["review_count"] > 0][["parent_asin", "year_month"]].drop_duplicates()
    first_review_month = (
        months_with_reviews
        .groupby("parent_asin", as_index=False)["year_month"]
        .min()
        .rename(columns={"year_month": "_first_review_month"})
    )

    grid = grid.merge(first_review_month, on="parent_asin", how="left")
    grid["is_warmup"] = grid["year_month"] == grid["_first_review_month"]
    grid.drop(columns=["_first_review_month"], inplace=True)
    return grid


def apply_shrinkage(grid_df: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    """Apply Empirical Bayes shrinkage toward the product-aspect lifetime prior.

    Shrinkage rule:
      - n = 0 mentions: masked (NaN)
      - n = 1 to 2 mentions: shrunk toward product-aspect lifetime mean: (n*x + k*prior) / (n + k)
      - n >= 3 mentions: keep raw sample mean
    """
    grid = grid_df.copy()
    if "is_warmup" not in grid.columns:
        grid = flag_warmup(grid)

    observed = grid[(grid["mention_count"] > 0) & (~grid["is_warmup"])].copy()
    if len(observed) == 0:
        observed = grid[grid["mention_count"] > 0].copy()

    # Lifetime mean weighted by mention count
    observed["_wx"] = observed["mention_count"] * observed["raw_mean_sentiment"]
    lifetime = (
        observed
        .groupby(["parent_asin", "canonical_aspect"], as_index=False)
        .agg(_sum_wx=("_wx", "sum"), _sum_n=("mention_count", "sum"))
    )
    lifetime["lifetime_mean"] = lifetime["_sum_wx"] / lifetime["_sum_n"]
    lifetime.drop(columns=["_sum_wx", "_sum_n"], inplace=True)

    grid = grid.merge(lifetime, on=["parent_asin", "canonical_aspect"], how="left")

    n = grid["mention_count"].astype(float)
    raw = grid["raw_mean_sentiment"]
    prior = grid["lifetime_mean"]
    shrunk_values = (n * raw + k * prior) / (n + k)

    grid["shrunk_score"] = np.where(
        grid["mention_count"] == 0,
        np.nan,
        np.where(grid["mention_count"] >= 3, raw, shrunk_values),
    )

    return grid


def normalize_mad(
    grid_df: pd.DataFrame,
    mad_floor: float = 0.1,
) -> pd.DataFrame:
    """Median / MAD normalization per product per aspect (outlier-robust z-score).

    Uses post-warmup observed months only. Scales by 1.4826 for normal consistency.
    Applies MAD floor = 0.1 (defaults to MAD=1.0) to prevent division by zero or
    extreme values that destabilize LSTM gradients.
    """
    grid = grid_df.copy()
    if "is_warmup" not in grid.columns:
        grid = flag_warmup(grid)
    if "shrunk_score" not in grid.columns:
        grid = apply_shrinkage(grid)

    fit_mask = (
        (~grid["is_warmup"])
        & (grid["mention_count"] > 0)
        & grid["shrunk_score"].notna()
    )

    medians = (
        grid.loc[fit_mask]
        .groupby(["parent_asin", "canonical_aspect"], as_index=False)
        .agg(aspect_median=("shrunk_score", "median"))
    )
    grid = grid.merge(medians, on=["parent_asin", "canonical_aspect"], how="left")

    grid["_abs_dev"] = (grid["shrunk_score"] - grid["aspect_median"]).abs()
    mad_df = (
        grid.loc[fit_mask]
        .groupby(["parent_asin", "canonical_aspect"], as_index=False)
        .agg(aspect_mad=("_abs_dev", "median"))
    )
    mad_df["aspect_mad"] = mad_df["aspect_mad"] * 1.4826  # Gaussian consistency
    grid = grid.merge(mad_df, on=["parent_asin", "canonical_aspect"], how="left")

    # Safeguard against tiny MADs
    grid["aspect_mad"] = np.where(grid["aspect_mad"] < mad_floor, 1.0, grid["aspect_mad"])

    grid["norm_score"] = np.where(
        grid["shrunk_score"].isna(),
        np.nan,
        (grid["shrunk_score"] - grid["aspect_median"]) / grid["aspect_mad"],
    )

    # Clip to [-5.0, 5.0] to protect gradient stability in LSTM-AE
    grid["norm_score"] = grid["norm_score"].clip(-5.0, 5.0)
    grid["aspect_mask"] = grid["aspect_mask"] & grid["norm_score"].notna()
    grid.drop(columns=["_abs_dev"], inplace=True)

    return grid


def normalize_mad_reference(
    grid_df: pd.DataFrame,
    n_baseline: int = 12,
    min_baseline: int = 6,
    mad_floor: float = 0.1,
) -> pd.DataFrame:
    """Reference-window MAD normalization for drift detection.

    Computes baseline statistics on the first n_baseline observed post-warmup months.
    This prevents ongoing drifts from inflating the scale and masking change points.
    """
    grid = grid_df.copy()
    if "is_warmup" not in grid.columns:
        grid = flag_warmup(grid)
    if "shrunk_score" not in grid.columns:
        grid = apply_shrinkage(grid)

    observed = grid[
        (~grid["is_warmup"])
        & (grid["mention_count"] > 0)
        & grid["shrunk_score"].notna()
    ].copy()

    observed["_obs_rank"] = observed.groupby(
        ["parent_asin", "canonical_aspect"]
    )["year_month"].rank(method="dense").astype(int)

    baseline = observed[observed["_obs_rank"] <= n_baseline].copy()

    ref_medians = (
        baseline
        .groupby(["parent_asin", "canonical_aspect"], as_index=False)
        .agg(
            ref_median=("shrunk_score", "median"),
            ref_n_baseline=("shrunk_score", "size"),
        )
    )
    grid = grid.merge(ref_medians, on=["parent_asin", "canonical_aspect"], how="left")

    baseline_with_med = baseline.merge(
        ref_medians[["parent_asin", "canonical_aspect", "ref_median"]],
        on=["parent_asin", "canonical_aspect"], how="left",
    )
    baseline_with_med["_ref_abs_dev"] = (
        baseline_with_med["shrunk_score"] - baseline_with_med["ref_median"]
    ).abs()

    ref_mads = (
        baseline_with_med
        .groupby(["parent_asin", "canonical_aspect"], as_index=False)
        .agg(ref_mad=("_ref_abs_dev", "median"))
    )
    ref_mads["ref_mad"] = ref_mads["ref_mad"] * 1.4826
    grid = grid.merge(ref_mads, on=["parent_asin", "canonical_aspect"], how="left")

    grid["ref_mad"] = np.where(grid["ref_mad"] < mad_floor, 1.0, grid["ref_mad"])
    grid["baseline_sufficient"] = grid["ref_n_baseline"] >= min_baseline

    grid["norm_score_ref"] = np.where(
        grid["shrunk_score"].isna() | ~grid["baseline_sufficient"],
        np.nan,
        (grid["shrunk_score"] - grid["ref_median"]) / grid["ref_mad"],
    )
    grid["norm_score_ref"] = grid["norm_score_ref"].clip(-5.0, 5.0)

    return grid


def build_star_rating_baseline(scaffold_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate star ratings into monthly product timeline as a naive benchmark."""
    scaffold = scaffold_df.copy()
    scaffold["year_month"] = _to_month_start(scaffold["timestamp"])
    scaffold["is_low_rating"] = scaffold["rating"].le(2).astype(float)

    product_month_grid = _build_product_month_grid(scaffold)

    star_monthly = (
        scaffold
        .groupby(["parent_asin", "year_month"], as_index=False)
        .agg(
            review_count=("review_pk", "nunique"),
            mean_rating=("rating", "mean"),
            pct_low_rating=("is_low_rating", "mean"),
            verified_ratio=("verified_purchase", "mean"),
        )
    )

    baseline = (
        product_month_grid
        .merge(star_monthly, on=["parent_asin", "year_month"], how="left")
        .sort_values(["parent_asin", "year_month"])
        .reset_index(drop=True)
    )

    baseline["review_count"] = baseline["review_count"].fillna(0).astype(int)
    baseline["log_review_count"] = np.log1p(baseline["review_count"].astype(float))

    return baseline
