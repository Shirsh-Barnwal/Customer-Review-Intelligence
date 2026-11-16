"""Export pipeline for dashboard artifacts and structured explainability bundles.

Builds and serializes multi-detector events, aspect sentiment deltas,
reconstruction error breakdowns, and customer review evidence directly into:
1. `dashboard/react/public/data/explainability.json`
2. `dashboard/react/public/data/alerts.json`
3. `results/explainability/summary.json`
4. Product-level JSON and Parquet reports
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from .drift_detection import (
    LSTMAutoencoder,
    SentimentWindowDataset,
    cusum_fuse,
    detect_cusum_product,
    detect_pelt_product,
    pelt_fuse,
    score_anomalies,
)
from .explainability import (
    compare_delta_recon,
    compute_event_delta,
    decompose_recon_error,
    rank_aspects,
    select_quotes,
)
from .utils import (
    CANONICAL_ASPECTS,
    DASHBOARD_DATA,
    NpEncoder,
    RESULTS_DIR,
)

__all__ = [
    "build_product_result",
    "export_results",
    "run_detectors",
    "analyze_product",
    "export_dashboard_artifacts",
]

# Standard detector hyperparameters matching project specifications
CUSUM_K = 0.5
CUSUM_H = 5.0
CUSUM_COOLDOWN = 2
PELT_BETA = 2.0
PELT_MIN_SIZE = 3
LSTM_SEQ_LEN = 6


def build_product_result(
    product: str,
    product_name: str,
    events_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    recon_df: pd.DataFrame | None,
    quotes_df: pd.DataFrame | None,
    monthly_df: pd.DataFrame,
    aspects: list[str],
    comparisons: dict | None = None,
) -> dict:
    """Build the comprehensive structured JSON document for a single product."""
    events = []
    prod_events = events_df[events_df["parent_asin"] == product] if "parent_asin" in events_df.columns else events_df

    for _, event_row in prod_events.iterrows():
        em = str(pd.Timestamp(event_row["event_month"]).date())
        detector = str(event_row.get("detector", "unknown"))
        em_period = pd.Timestamp(em).to_period("M")

        if not deltas_df.empty and "event_month" in deltas_df.columns:
            event_deltas = deltas_df[
                (deltas_df["parent_asin"] == product)
                & (pd.to_datetime(deltas_df["event_month"]).dt.to_period("M") == em_period)
            ]
        else:
            event_deltas = pd.DataFrame()

        top_aspects_list = []
        if not event_deltas.empty:
            sorted_deltas = event_deltas.sort_values("abs_delta", ascending=False)
            for _, d in sorted_deltas.head(5).iterrows():
                entry = {
                    "aspect": d["canonical_aspect"],
                    "delta": float(d["delta"]) if pd.notna(d["delta"]) else None,
                    "direction": d.get("direction", "unknown"),
                    "confidence": d["confidence"],
                }
                if recon_df is not None and not recon_df.empty:
                    recon_match = recon_df[
                        (recon_df["parent_asin"] == product)
                        & (pd.to_datetime(recon_df["event_month"]).dt.to_period("M") == em_period)
                        & (recon_df["canonical_aspect"] == d["canonical_aspect"])
                    ]
                    if not recon_match.empty:
                        entry["recon_contrib"] = float(recon_match.iloc[0]["contribution_pct"])
                top_aspects_list.append(entry)

        recon_breakdown = []
        if recon_df is not None and not recon_df.empty:
            event_recon = recon_df[
                (recon_df["parent_asin"] == product)
                & (pd.to_datetime(recon_df["event_month"]).dt.to_period("M") == em_period)
            ].sort_values("contribution_pct", ascending=False)
            for _, r in event_recon.iterrows():
                recon_breakdown.append({
                    "aspect": r["canonical_aspect"],
                    "contribution_pct": float(r["contribution_pct"]),
                })

        event_quotes = []
        if quotes_df is not None and not quotes_df.empty and "event_month" in quotes_df.columns:
            eq = quotes_df[
                (quotes_df["parent_asin"] == product)
                & (quotes_df["event_month"] == em)
            ]
            for _, q in eq.iterrows():
                event_quotes.append({
                    "text": q["review_text_snippet"],
                    "aspect": q["canonical_aspect"],
                    "sentiment": float(q["sentiment_score"]),
                    "stars": int(q["stars"]),
                    "month": q["review_month"],
                })

        agreement = "N/A"
        if comparisons and em in comparisons:
            agreement = str(comparisons[em].get("agreement", "N/A"))

        events.append({
            "event_month": em,
            "detector": detector,
            "top_aspects": top_aspects_list,
            "recon_breakdown": recon_breakdown,
            "agreement": agreement,
            "quotes": event_quotes,
        })

    prod_monthly = monthly_df[monthly_df["parent_asin"] == product].copy()
    prod_monthly["year_month"] = pd.to_datetime(prod_monthly["year_month"])
    timeline = []

    for m in sorted(prod_monthly["year_month"].unique()):
        month_data = prod_monthly[prod_monthly["year_month"] == m]
        aspect_scores = {}
        for _, row in month_data.iterrows():
            asp = row["canonical_aspect"]
            if row["aspect_mask"] and pd.notna(row.get("raw_mean_sentiment")):
                aspect_scores[asp] = round(float(row["raw_mean_sentiment"]), 4)
        timeline.append({
            "month": str(m.date()),
            "aspects": aspect_scores,
        })

    action_items = []
    if not deltas_df.empty:
        prod_deltas = deltas_df[deltas_df["parent_asin"] == product]
        for asp in aspects:
            asp_deltas = prod_deltas[prod_deltas["canonical_aspect"] == asp]
            if asp_deltas.empty:
                continue
            delta_vals = pd.to_numeric(asp_deltas["delta"], errors="coerce")
            min_val = delta_vals.min()
            if pd.isna(min_val) or min_val >= 0:
                continue
            worst = asp_deltas.loc[delta_vals.idxmin()]
            if pd.notna(worst["delta"]) and worst["delta"] < -0.1:
                action_items.append({
                    "aspect": asp,
                    "severity": round(abs(float(worst["delta"])), 4),
                    "direction": "deterioration",
                    "evidence_strength": worst["confidence"],
                })

    action_items.sort(key=lambda x: x["severity"], reverse=True)

    return {
        "parent_asin": product,
        "product_name": product_name,
        "n_events": len(events),
        "events": events,
        "timeline": timeline,
        "action_items": action_items[:5],
    }


def run_detectors(
    asin: str,
    monthly_df: pd.DataFrame,
    model: LSTMAutoencoder | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Execute CUSUM, PELT, and LSTM-AE detectors on a single product time series."""
    prod_df = monthly_df[monthly_df["parent_asin"] == asin].copy()
    if prod_df.empty:
        return pd.DataFrame(columns=["parent_asin", "event_month", "detector"])

    events = []

    # CUSUM sequential control chart
    cusum_det = detect_cusum_product(prod_df, CANONICAL_ASPECTS, k=CUSUM_K, h=CUSUM_H, cooldown=CUSUM_COOLDOWN)
    cusum_fused = cusum_fuse(cusum_det)
    for _, row in cusum_fused.iterrows():
        events.append({"parent_asin": asin, "event_month": row["event_month"], "detector": "CUSUM"})

    # PELT changepoint detection
    pelt_det = detect_pelt_product(prod_df, CANONICAL_ASPECTS, beta=PELT_BETA, min_size=PELT_MIN_SIZE)
    pelt_fused = pelt_fuse(pelt_det)
    for _, row in pelt_fused.iterrows():
        events.append({"parent_asin": asin, "event_month": row["event_month"], "detector": "PELT"})

    # LSTM Autoencoder anomaly scoring
    if model is not None:
        dataset = SentimentWindowDataset({asin: prod_df}, CANONICAL_ASPECTS, seq_len=LSTM_SEQ_LEN, stride=1)
        if len(dataset) > 0:
            ae_scores = score_anomalies(model, dataset, device=device)
            anomalies = ae_scores[ae_scores["is_anomaly"]].sort_values("last_month")
            if len(anomalies) > 0:
                ae_months = sorted(anomalies["last_month"].unique())
                clusters = [[ae_months[0]]]
                for m in ae_months[1:]:
                    prev = clusters[-1][-1]
                    gap = (m.year - prev.year) * 12 + (m.month - prev.month)
                    if gap <= 2:
                        clusters[-1].append(m)
                    else:
                        clusters.append([m])
                for cluster in clusters:
                    events.append({"parent_asin": asin, "event_month": cluster[0], "detector": "LSTM-AE"})

    return pd.DataFrame(events)


def analyze_product(
    asin: str,
    name: str,
    events_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    absa_df: pd.DataFrame,
    scaffold_df: pd.DataFrame,
    model: LSTMAutoencoder | None = None,
    device: str = "cpu",
) -> tuple[dict | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform complete explainability analysis on an individual product."""
    prod_events = events_df[events_df["parent_asin"] == asin]
    if prod_events.empty:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    unique_months = sorted(prod_events["event_month"].unique())
    all_deltas, all_recon, all_quotes = [], [], []
    comparisons = {}

    for ev_month in unique_months:
        deltas = rank_aspects(compute_event_delta(monthly_df, asin, ev_month, CANONICAL_ASPECTS, window=3))
        deltas["parent_asin"] = asin
        deltas["event_month"] = ev_month
        all_deltas.append(deltas)

        if model is not None:
            recon = decompose_recon_error(model, monthly_df, asin, ev_month, CANONICAL_ASPECTS, seq_len=LSTM_SEQ_LEN, device=device)
            recon["parent_asin"] = asin
            recon["event_month"] = ev_month
            all_recon.append(recon)
            comparison = compare_delta_recon(deltas, recon)
            comparisons[str(pd.Timestamp(ev_month).date())] = comparison

        top_aspects = deltas.head(3)["canonical_aspect"].tolist()
        asp_delta_dict = dict(zip(deltas["canonical_aspect"], deltas["delta"]))
        quotes = select_quotes(reviews_df, absa_df, scaffold_df, asin, ev_month, top_aspects, window=3, n_quotes=6, aspect_deltas=asp_delta_dict)
        if not quotes.empty:
            all_quotes.append(quotes)

    all_deltas_df = pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.DataFrame()
    all_recon_df = pd.concat(all_recon, ignore_index=True) if all_recon else pd.DataFrame()
    all_quotes_df = pd.concat(all_quotes, ignore_index=True) if all_quotes else pd.DataFrame()

    result = build_product_result(
        product=asin,
        product_name=name,
        events_df=prod_events,
        deltas_df=all_deltas_df,
        recon_df=all_recon_df,
        quotes_df=all_quotes_df,
        monthly_df=monthly_df,
        aspects=CANONICAL_ASPECTS,
        comparisons=comparisons,
    )

    return result, all_deltas_df, all_recon_df, all_quotes_df


def export_results(
    all_results: list[dict],
    deltas_df: pd.DataFrame,
    events_df: pd.DataFrame,
    recon_df: pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
    dashboard_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Write explainability artifacts and dashboard JSON files to disk."""
    out_dir = Path(output_dir) if output_dir else RESULTS_DIR / "explainability"
    dash_dir = Path(dashboard_dir) if dashboard_dir else DASHBOARD_DATA

    out_dir.mkdir(parents=True, exist_ok=True)
    dash_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    paths = {}
    for res in all_results:
        asin = res["parent_asin"]
        p = out_dir / f"product_{asin}.json"
        p.write_text(json.dumps(res, indent=2, cls=NpEncoder), encoding="utf-8")
        paths[f"product_{asin}"] = p

    summary = {
        "n_products": len(all_results),
        "products": [
            {
                "parent_asin": r["parent_asin"],
                "product_name": r["product_name"],
                "n_events": r["n_events"],
                "top_action": r["action_items"][0] if r["action_items"] else None,
            }
            for r in all_results
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, cls=NpEncoder), encoding="utf-8")
    paths["summary"] = summary_path

    # Export main bundle for React dashboard
    dash_path = dash_dir / "explainability.json"
    dash_path.write_text(json.dumps({"products": all_results, "summary": summary}, indent=2, cls=NpEncoder), encoding="utf-8")
    paths["dashboard_explainability"] = dash_path

    return paths


def export_dashboard_artifacts(
    case_studies: dict[str, str],
    monthly_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    absa_df: pd.DataFrame,
    scaffold_df: pd.DataFrame,
    model: LSTMAutoencoder | None = None,
    device: str = "cpu",
) -> None:
    """Run full detection and export pipeline across case study smartphone products."""
    all_results = []
    all_deltas, all_events, all_recons = [], [], []

    for asin, name in case_studies.items():
        events_df = run_detectors(asin, monthly_df, model=model, device=device)
        res, deltas, recon, quotes = analyze_product(
            asin=asin,
            name=name,
            events_df=events_df,
            monthly_df=monthly_df,
            reviews_df=reviews_df,
            absa_df=absa_df,
            scaffold_df=scaffold_df,
            model=model,
            device=device,
        )
        if res:
            all_results.append(res)
            if not deltas.empty:
                all_deltas.append(deltas)
            if not events_df.empty:
                all_events.append(events_df)
            if not recon.empty:
                all_recons.append(recon)

    d_df = pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.DataFrame()
    e_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    r_df = pd.concat(all_recons, ignore_index=True) if all_recons else pd.DataFrame()

    export_results(all_results, d_df, e_df, r_df)
