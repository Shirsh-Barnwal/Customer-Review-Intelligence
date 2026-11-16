"""PELT changepoint detection compatibility shim."""
from ..drift_detection import (
    forward_fill_series,
    detect_pelt,
    detect_pelt_product,
    pelt_fuse as fuse_events,
)
