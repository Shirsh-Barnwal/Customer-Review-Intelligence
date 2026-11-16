"""CUSUM detection compatibility shim."""
from ..drift_detection import (
    detect_cusum,
    detect_cusum_product,
    cusum_fuse as fuse_events,
)
