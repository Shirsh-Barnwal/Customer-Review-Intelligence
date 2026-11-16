"""Customer Review Intelligence (SentinelDrift) Package.

A complete pipeline for Aspect-Based Sentiment Analysis (PyABSA FAST-LSA-T-V2),
monthly temporal aggregation, multi-method drift detection (CUSUM, PELT, LSTM-AE),
and root-cause explainability with an interactive React dashboard.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except Exception:
    pass

from .utils import (
    ROOT,
    CANONICAL_ASPECTS,
    ASPECT_COLORS,
    ASPECT_SYNONYMS,
    DATA_RAW,
    DATA_PROCESSED,
    DATA_TIMESERIES,
    DATA_LABELS,
    MODELS_DIR,
    RESULTS_DIR,
    DASHBOARD_DATA,
    set_seed,
    NpEncoder,
)
from .data_processing import (
    make_review_pk,
    clean_reviews,
    build_monthly_aspect_grid,
    flag_warmup,
    apply_shrinkage,
    normalize_mad,
    normalize_mad_reference,
    build_star_rating_baseline,
)
from .absa_models import (
    detect_aspects,
    inject_aspect_tags,
    predict_single,
    predict_batch,
    TfidfBaseline,
    build_pair_dataset,
    evaluate_absa_5fold,
)
from .drift_detection import (
    detect_cusum,
    detect_cusum_product,
    cusum_fuse,
    detect_pelt,
    detect_pelt_product,
    pelt_fuse,
    SentimentWindowDataset,
    LSTMAutoencoder,
    masked_mse_loss,
    train_model,
    score_anomalies,
    detect_lstm_ae_product,
)
from .explainability import (
    compute_event_delta,
    rank_aspects,
    compute_all_events,
    bootstrap_delta_ci,
    random_window_null,
    decompose_recon_error,
    compare_delta_recon,
    decompose_all_events,
    compute_attributions,
    render_heatmap,
    select_quotes,
    keyword_evidence,
    NEXUS_6P_KEYWORDS,
)
from .export_pipeline import (
    build_product_result,
    export_results,
    run_detectors,
    analyze_product,
    export_dashboard_artifacts,
)

__all__ = [
    "ROOT",
    "CANONICAL_ASPECTS",
    "ASPECT_COLORS",
    "ASPECT_SYNONYMS",
    "set_seed",
    "NpEncoder",
    "make_review_pk",
    "clean_reviews",
    "build_monthly_aspect_grid",
    "flag_warmup",
    "apply_shrinkage",
    "normalize_mad",
    "normalize_mad_reference",
    "build_star_rating_baseline",
    "detect_aspects",
    "inject_aspect_tags",
    "predict_single",
    "predict_batch",
    "TfidfBaseline",
    "build_pair_dataset",
    "evaluate_absa_5fold",
    "detect_cusum",
    "detect_cusum_product",
    "cusum_fuse",
    "detect_pelt",
    "detect_pelt_product",
    "pelt_fuse",
    "SentimentWindowDataset",
    "LSTMAutoencoder",
    "masked_mse_loss",
    "train_model",
    "score_anomalies",
    "detect_lstm_ae_product",
    "compute_event_delta",
    "rank_aspects",
    "compute_all_events",
    "bootstrap_delta_ci",
    "random_window_null",
    "decompose_recon_error",
    "compare_delta_recon",
    "decompose_all_events",
    "compute_attributions",
    "render_heatmap",
    "select_quotes",
    "keyword_evidence",
    "NEXUS_6P_KEYWORDS",
    "build_product_result",
    "export_results",
    "run_detectors",
    "analyze_product",
    "export_dashboard_artifacts",
]
