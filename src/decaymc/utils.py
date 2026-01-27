"""
Small utilities used for validation and analysis.
"""

from __future__ import annotations
import numpy as np


def l1_distance_hist(a: np.ndarray, b: np.ndarray, bins: int, range_: tuple[float, float]) -> float:
    """
    L1 distance between two 1D densities estimated by histograms with same bins/range.
    """
    ha, edges = np.histogram(a, bins=bins, range=range_, density=True)
    hb, _ = np.histogram(b, bins=bins, range=range_, density=True)
    dx = edges[1] - edges[0]
    return float(np.sum(np.abs(ha - hb)) * dx)

