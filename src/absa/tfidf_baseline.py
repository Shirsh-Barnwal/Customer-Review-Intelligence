"""TF-IDF baseline compatibility shim."""
from ..absa_models import (
    CANONICAL_ASPECTS,
    BASELINE_LABELS as LABELS,
    TfidfBaseline,
    build_pair_dataset,
    evaluate_absa_5fold,
)
