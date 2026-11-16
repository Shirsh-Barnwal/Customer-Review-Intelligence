"""PyABSA runner compatibility shim."""
from ..absa_models import (
    ASPECT_SYNONYMS,
    _KEYWORD_TO_ASPECT,
    make_review_pk,
    detect_aspects,
    inject_aspect_tags,
    predict_single,
    predict_batch,
    _extract_aspect_context,
    _load_classifier,
    _normalize_pyabsa_output,
)
