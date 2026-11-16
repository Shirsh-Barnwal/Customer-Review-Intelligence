"""Data loading compatibility shim."""
from ..data_processing import clean_reviews

def load_raw(path):
    raise NotImplementedError("Raw Amazon dataset is not committed; use data/processed/")
