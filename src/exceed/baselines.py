"""Thresholding baselines for one-class anomaly segmentation. All rules see the same inputs as EXCEED: score maps of n nominal calibration images
(shape (n, H, W)) and the test score map (H, W). The oracle uses test labels and is a reference only.
"""
from __future__ import annotations

import numpy as np

from .evalues import bh, by


def perlocation_pvalues(test: np.ndarray, cal: np.ndarray) -> np.ndarray:
    """p(u) = (1 + #{i : S_i(u) >= S_0(u)}) / (n + 1). Valid under image exchangeability,
    but bounded below by 1/(n+1) (resolution barrier, Prop. 2)."""
    n = cal.shape[0]
    return (1.0 + (cal >= test[None]).sum(axis=0)) / (n + 1.0)


def pooled_pvalues(test: np.ndarray, cal_sorted: np.ndarray) -> np.ndarray:
    """p(u) = (1 + #{calibration units with score >= S_0(u)}) / (n m + 1), pooling all
    units of all calibration images (cal_sorted: ascending 1-D array of n*m scores)."""
    N = cal_sorted.size
    ge = N - np.searchsorted(cal_sorted, test.ravel(), side="left")
    return ((1.0 + ge) / (N + 1.0)).reshape(test.shape)


def perlocation_bh(test, cal, alpha):
    return bh(perlocation_pvalues(test, cal), alpha)


def pooled_bh(test, cal_sorted, alpha):
    return bh(pooled_pvalues(test, cal_sorted), alpha)


def pooled_by(test, cal_sorted, alpha):
    return by(pooled_pvalues(test, cal_sorted), alpha)


def max_nominal_threshold(cal_max: np.ndarray, alpha: float) -> float:
    """Image-level FWER rule: t = ceil((1-alpha)(n+1))-th smallest per-image maximum.
    A nominal test image has P(max >= t) <= alpha. Returns +inf if n is too small."""
    n = cal_max.size
    k = int(np.ceil((1 - alpha) * (n + 1)))
    if k > n:
        return np.inf
    return float(np.sort(cal_max)[k - 1])


def percentile_threshold(cal_sorted: np.ndarray, q: float) -> float:
    """Common practice: a high quantile q of pooled nominal unit scores (no guarantee)."""
    return float(np.quantile(cal_sorted, q))


def crc_fpr_threshold(cal_counts_grid: np.ndarray, grid: np.ndarray, m: int, gamma: float) -> float:
    """Conformal risk control of the expected fraction of flagged units in a nominal
    image (unit-level false-positive rate): smallest t on an increasing grid with
    (sum_i N_i(t)/m + 1) / (n + 1) <= gamma. cal_counts_grid: (n, len(grid))."""
    n = cal_counts_grid.shape[0]
    risk = (cal_counts_grid.sum(axis=0) / m + 1.0) / (n + 1.0)
    ok = np.nonzero(risk <= gamma)[0]
    return float(grid[ok.min()]) if ok.size else np.inf


def oracle_f1_threshold(test_scores: np.ndarray, test_masks: np.ndarray, grid: np.ndarray) -> float:
    """Threshold maximising pooled pixel F1 on the TEST set (uses labels; reference)."""
    s = test_scores.ravel()
    y = test_masks.ravel().astype(bool)
    best, best_t = -1.0, float(grid[0])
    for t in grid:
        r = s >= t
        tp = np.count_nonzero(r & y)
        fp = np.count_nonzero(r & ~y)
        fn = np.count_nonzero(~r & y)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t
