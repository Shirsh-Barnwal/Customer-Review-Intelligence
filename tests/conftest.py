"""Pytest configuration and Windows DLL initialization."""
import os
import sys

# Allow multiple OpenMP runtimes on Windows to prevent WinError 1114 conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except Exception:
    pass
