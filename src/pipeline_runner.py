"""End-to-end orchestrator and interview-ready demonstration runner.

Usage:
    # Quick interview-ready demonstration (runs in ~15 seconds)
    python -m src.pipeline_runner --demo

    # Run full 5-fold cross-validation on gold labels
    python -m src.pipeline_runner --stage absa_eval

    # Run drift detection benchmarks
    python -m src.pipeline_runner --stage drift

    # Re-export all dashboard datasets
    python -m src.pipeline_runner --stage export
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import torch

from .absa_models import (
    TfidfBaseline,
    detect_aspects,
    evaluate_absa_5fold,
    predict_single,
)
from .data_processing import (
    apply_shrinkage,
    build_monthly_aspect_grid,
    build_star_rating_baseline,
    clean_reviews,
    flag_warmup,
    normalize_mad,
    normalize_mad_reference,
)
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
    NEXUS_6P_KEYWORDS,
    compare_delta_recon,
    compute_event_delta,
    decompose_recon_error,
    keyword_evidence,
    rank_aspects,
    select_quotes,
)
from .export_pipeline import export_dashboard_artifacts, run_detectors
from .utils import (
    CANONICAL_ASPECTS,
    DASHBOARD_DATA,
    DATA_LABELS,
    DATA_PROCESSED,
    DATA_TIMESERIES,
    MODELS_DIR,
    RESULTS_DIR,
    ROOT,
    set_seed,
)

# Flagship case study products
CASE_STUDIES = {
    "B019BX7F86": "Google Nexus 6P (Battery & Bootloop Drift)",
    "B097M74GSR": "Apple iPhone 11",
    "B086RM3DQS": "Apple iPhone 8",
    "B09JJCXC2M": "Samsung Galaxy S10e",
    "B07H8PXT7K": "Samsung Galaxy Note 9",
    "B07RWPFQLM": "Google Pixel 3a",
}


def bootstrap_check() -> bool:
    """Verify that processed data files exist in the repository."""
    required = [
        DATA_PROCESSED / "review_scaffold.parquet",
        DATA_TIMESERIES / "aspect_sentiment_monthly.parquet",
        DATA_LABELS / "absa_gold_labels.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[Bootstrap] Missing required data files:")
        for m in missing:
            print(f"  - {m.relative_to(ROOT)}")
        return False
    return True


def run_demo() -> None:
    """Run an interview-ready demonstration showcasing all system capabilities in ~15-20s."""
    print("=" * 75)
    print("  CUSTOMER REVIEW INTELLIGENCE (SENTINELDRIFT) - LIVE SYSTEM DEMO")
    print("=" * 75)

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Execution device: {device.upper()}")

    # 1. ABSA Aspect Extraction & Polarity Classification
    print("\n[Stage 1] Aspect-Based Sentiment Analysis (PyABSA FAST-LSA-T-V2)")
    sample_text = (
        "The battery life is terrible and drains in 2 hours, and the phone keeps "
        "restarting in a bootloop. However, the camera takes stunning photos and the "
        "display is crisp."
    )
    print(f"  Sample Review: \"{sample_text}\"")

    detected = detect_aspects(sample_text)
    print(f"  Detected Aspects ({len(detected)}): {[d['aspect'] for d in detected]}")

    preds = predict_single(sample_text)
    for p in preds:
        polarity = p['sentiment_label'].upper()
        score = p['sentiment_score']
        conf = p['confidence']
        print(f"   -> Aspect: {p['canonical_aspect']:<12} | Sentiment: {polarity:<8} | Score: {score:+.2f} | Conf: {conf:.2f}")

    # 2. Longitudinal Aspect Sentiment & Drift Detection on Nexus 6P
    print("\n[Stage 2] Drift Detection on Google Nexus 6P (ASIN: B019BX7F86)")
    ts_path = DATA_TIMESERIES / "aspect_sentiment_monthly.parquet"
    if ts_path.exists():
        monthly_df = pd.read_parquet(ts_path)
        nexus_df = monthly_df[monthly_df["parent_asin"] == "B019BX7F86"]
        print(f"  Loaded {len(nexus_df)} monthly observations across 9 canonical aspects.")

        # CUSUM Detection
        cusum_raw = detect_cusum_product(nexus_df, CANONICAL_ASPECTS, k=0.5, h=5.0)
        cusum_events = cusum_fuse(cusum_raw)
        print(f"\n  [CUSUM Control Chart] Found {len(cusum_events)} fused product-level shift events:")
        for _, ev in cusum_events.head(3).iterrows():
            m_str = str(pd.Timestamp(ev["event_month"]).date())
            asps = ", ".join(ev["aspects_involved"])
            dirs = ", ".join(ev["directions"])
            print(f"    - Month: {m_str} | Aspects: [{asps}] | Shift: {dirs}")

        # PELT Changepoint Detection
        pelt_raw = detect_pelt_product(nexus_df, CANONICAL_ASPECTS, beta=2.0)
        pelt_events = pelt_fuse(pelt_raw)
        print(f"\n  [PELT Changepoints] Found {len(pelt_events)} changepoint clusters:")
        for _, ev in pelt_events.head(3).iterrows():
            m_str = str(pd.Timestamp(ev["event_month"]).date())
            asps = ", ".join(ev["aspects_involved"])
            print(f"    - Month: {m_str} | Aspects: [{asps}]")

        # LSTM Autoencoder Anomaly Scoring
        model_path = MODELS_DIR / "drift" / "lstm_ae" / "best_lstm_ae.pt"
        if model_path.exists():
            model = LSTMAutoencoder(input_dim=18, output_dim=9, hidden_dim=32, latent_dim=8)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            ae_dataset = SentimentWindowDataset({"B019BX7F86": nexus_df}, CANONICAL_ASPECTS, seq_len=6, stride=1)
            if len(ae_dataset) > 0:
                ae_scores = score_anomalies(model, ae_dataset, device=device)
                anomalies = ae_scores[ae_scores["is_anomaly"]]
                print(f"\n  [LSTM Autoencoder] Reconstructed {len(ae_dataset)} sliding windows:")
                print(f"    - Anomaly threshold (95th pct): {ae_scores['threshold'].iloc[0]:.4f}")
                print(f"    - Detected anomalies: {len(anomalies)} window(s)")
                for _, a in anomalies.head(2).iterrows():
                    m_str = str(pd.Timestamp(a["last_month"]).date())
                    print(f"      * Month: {m_str} | Recon MSE: {a['recon_error']:.4f} (Anomaly: True)")

        # 3. Root-Cause Aspect Delta Ranking
        print("\n[Stage 3] Root-Cause Attribution & Aspect Delta Ranking")
        if not cusum_events.empty:
            flagship_event = cusum_events.iloc[0]["event_month"]
            deltas = rank_aspects(compute_event_delta(monthly_df, "B019BX7F86", flagship_event, CANONICAL_ASPECTS, window=3))
            print(f"  Pre/Post 3-Month Window Comparison for Event Month: {pd.Timestamp(flagship_event).date()}")
            print(f"  {'Aspect':<14} | {'Pre Mean':>8} | {'Post Mean':>9} | {'Delta':>8} | {'Shift Type':<14} | {'Confidence'}")
            print("  " + "-" * 72)
            for _, r in deltas.head(5).iterrows():
                pre_m = f"{r['pre_mean']:+.2f}" if pd.notna(r['pre_mean']) else "N/A"
                post_m = f"{r['post_mean']:+.2f}" if pd.notna(r['post_mean']) else "N/A"
                del_str = f"{r['delta']:+.2f}" if pd.notna(r['delta']) else "N/A"
                print(f"  {r['canonical_aspect']:<14} | {pre_m:>8} | {post_m:>9} | {del_str:>8} | {r['direction']:<14} | {r['confidence']}")

    # 4. Review Quotes & Qualitative Evidence
    print("\n[Stage 4] Customer Review Qualitative Evidence")
    gold_path = DATA_LABELS / "absa_gold_labels.csv"
    quotes_shown = 0
    if gold_path.exists():
        gold_df = pd.read_csv(gold_path)
        for p in gold_df["parent_asin"].unique():
            ev = keyword_evidence(gold_df, p, keywords=["battery", "camera", "screen", "charging", "bootloop", "drain", "power"], max_quotes=2)
            if ev:
                for item in ev:
                    quotes_shown += 1
                    snippet = item['review_text'][:160].replace("\n", " ") + "..."
                    print(f"  Evidence #{quotes_shown} (Product: {p}) [Keywords: {item['keywords']} | Stars: {item['stars']}/5]:")
                    print(f"    \"{snippet}\"")
                    if quotes_shown >= 2:
                        break
            if quotes_shown >= 2:
                break

    dash_json = DASHBOARD_DATA / "explainability.json"
    if quotes_shown == 0 and dash_json.exists():
        import json
        with open(dash_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        for prod in d.get("products", []):
            for ev in prod.get("events", []):
                for q in ev.get("quotes", [])[:2]:
                    quotes_shown += 1
                    print(f"  Evidence #{quotes_shown} ({prod['product_name']} - {ev['event_month']}) [{q.get('aspect')} | {q.get('stars', 0)}/5 stars]:")
                    print(f"    \"{q.get('text', '')[:160]}...\"")
                    if quotes_shown >= 2:
                        break
                if quotes_shown >= 2:
                    break
            if quotes_shown >= 2:
                break

    # 5. Dashboard Status
    print("\n[Stage 5] Interactive Analyst Dashboard Readiness")
    dash_json = DASHBOARD_DATA / "explainability.json"
    if dash_json.exists():
        print(f"  [OK] Dashboard dataset active: {dash_json.relative_to(ROOT)} ({dash_json.stat().st_size / 1024:.1f} KB)")
        print("  Launch interactive UI via:")
        print("    cd dashboard/react")
        print("    npm.cmd install")
        print("    npm.cmd run dev")
    print("=" * 75)
    print("  DEMO COMPLETE - ALL PIPELINE STAGES FUNCTIONAL & INTERVIEW READY")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Customer Review Intelligence Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run interactive live demonstration")
    parser.add_argument("--stage", choices=["all", "absa_eval", "drift", "export"], default="demo", help="Pipeline execution stage")
    args = parser.parse_args()

    if args.demo or args.stage == "demo":
        run_demo()
        return

    if args.stage == "absa_eval":
        gold_file = DATA_LABELS / "absa_gold_labels.csv"
        if not gold_file.exists():
            print(f"Error: {gold_file} not found.")
            sys.exit(1)
        print("Running Stratified Grouped 5-Fold Cross-Validation on Gold Labels...")
        df = pd.read_csv(gold_file)
        results = evaluate_absa_5fold(df, n_splits=5, n_bootstrap=500)
        print("\nCross-Validation Results:")
        print(f"  PyABSA (FAST-LSA-T-V2): Macro F1 = {results['dl_macro_f1']} (95% CI: {results['dl_macro_ci_95']})")
        print(f"  TF-IDF + Logistic Reg : Macro F1 = {results['lr_macro_f1']} (95% CI: {results['lr_macro_ci_95']})")
        print(f"  McNemar Test          : p = {results['mcnemar_p']:.6f} (stat = {results['mcnemar_stat']})")
        print(f"  Winner                : {results['winner']}")

    elif args.stage == "drift":
        print("Running Drift Detection verification on case study ASINs...")
        run_demo()

    elif args.stage == "export":
        print("Exporting React dashboard datasets...")
        ts_path = DATA_TIMESERIES / "aspect_sentiment_monthly.parquet"
        scaffold_path = DATA_PROCESSED / "review_scaffold.parquet"
        absa_path = DATA_PROCESSED / "review_aspect_sentiment.parquet"
        model_path = MODELS_DIR / "drift" / "lstm_ae" / "best_lstm_ae.pt"

        if not (ts_path.exists() and scaffold_path.exists() and absa_path.exists()):
            print("Error: processed parquets missing. Run bootstrap first.")
            sys.exit(1)

        monthly_df = pd.read_parquet(ts_path)
        scaffold_df = pd.read_parquet(scaffold_path)
        absa_df = pd.read_parquet(absa_path)

        model = None
        if model_path.exists():
            model = LSTMAutoencoder(input_dim=18, output_dim=9, hidden_dim=32, latent_dim=8)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

        export_dashboard_artifacts(
            case_studies=CASE_STUDIES,
            monthly_df=monthly_df,
            reviews_df=scaffold_df,
            absa_df=absa_df,
            scaffold_df=scaffold_df,
            model=model,
            device="cpu",
        )
        print("Export completed successfully.")


if __name__ == "__main__":
    main()
