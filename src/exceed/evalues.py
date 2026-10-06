"""Compound e-values from image exchangeability and the e-BH procedure.

Notation (paper Sec. 4). Image j in {0 (test), 1..n (calibration)} has m units with
scores S_j(u). For a fixed nonnegative function g, the g-mass of image j is
G_j = sum_u g(S_j(u)). EXCEED assigns to test unit u the value

    e(u) = (n + 1) * m * g(S_0(u)) / (G_0 + sum_{i=1}^n G_i),        0/0 := 0.

For g = sum_k w_k 1{s >= t_k} this is the threshold-mixture form used in the paper.
Under image exchangeability the null e-values sum to at most m in expectation
(compound e-values), so e-BH controls the unit-level FDR under arbitrary
within-image dependence (Theorem 1).
"""
from __future__ import annotations

import numpy as np


def exceedance_counts(scores: np.ndarray, thresholds) -> np.ndarray:
    """Counts N_j(t_k) = #{u : S_j(u) >= t_k}.

    scores: array of shape (n_images, *unit_shape); thresholds: 1-D array of K values.
    Returns an integer array of shape (n_images, K).
    """
    s = np.asarray(scores).reshape(len(scores), -1)
    t = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    out = np.empty((s.shape[0], t.size), dtype=np.int64)
    for k, tk in enumerate(t):
        out[:, k] = (s >= tk).sum(axis=1)
    return out


def threshold_evalues(test_scores: np.ndarray, cal_counts: np.ndarray, thresholds,
                      weights=None) -> np.ndarray:
    """EXCEED e-values for one test image with the threshold-mixture g.

    test_scores: array of shape unit_shape (m units).
    cal_counts: (n, K) exceedance counts of the n calibration images at the K thresholds.
    weights: K nonnegative weights summing to one (default: uniform).
    """
    s = np.asarray(test_scores, dtype=np.float64)
    t = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    cal_counts = np.asarray(cal_counts).reshape(-1, t.size)
    n = cal_counts.shape[0]
    m = s.size
    w = np.full(t.size, 1.0 / t.size) if weights is None else np.asarray(weights, float)
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    e = np.zeros_like(s)
    cal_tot = cal_counts.sum(axis=0)
    for k, tk in enumerate(t):
        hit = s >= tk
        n0 = int(hit.sum())
        denom = n0 + int(cal_tot[k])
        if n0 == 0 or denom == 0:
            continue
        e[hit] += w[k] * (n + 1) * m / denom
    return e


def gmass_evalues(test_scores: np.ndarray, cal_gmass: np.ndarray, g) -> np.ndarray:
    """EXCEED e-values for a general nonnegative g (cal_gmass: G_i for i = 1..n)."""
    gs = np.asarray(g(np.asarray(test_scores, dtype=np.float64)), dtype=np.float64)
    if np.any(gs < 0):
        raise ValueError("g must be nonnegative")
    n = int(np.size(cal_gmass))
    denom = gs.sum() + float(np.sum(cal_gmass))
    if denom <= 0:
        return np.zeros_like(gs)
    return (n + 1) * gs.size * gs / denom


def ebh(e: np.ndarray, alpha: float, m: int | None = None) -> np.ndarray:
    """e-BH (Wang & Ramdas, 2022): reject the k* largest e-values,
    k* = max{k : e_(k) >= m / (alpha k)}. Returns a boolean mask shaped like e."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    flat = np.asarray(e, dtype=np.float64).ravel()
    m = flat.size if m is None else int(m)
    order = np.argsort(-flat, kind="stable")
    es = flat[order]
    k = np.arange(1, flat.size + 1)
    ok = np.nonzero(es >= m / (alpha * k))[0]
    rej = np.zeros(flat.size, dtype=bool)
    if ok.size:
        rej[order[: ok.max() + 1]] = True
    return rej.reshape(np.shape(e))


def bh(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg on an array of p-values (mask of rejections)."""
    flat = np.asarray(p, dtype=np.float64).ravel()
    m = flat.size
    order = np.argsort(flat, kind="stable")
    ps = flat[order]
    ok = np.nonzero(ps <= alpha * np.arange(1, m + 1) / m)[0]
    rej = np.zeros(m, dtype=bool)
    if ok.size:
        rej[order[: ok.max() + 1]] = True
    return rej.reshape(np.shape(p))


def by(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Yekutieli (valid under arbitrary dependence)."""
    m = np.size(p)
    return bh(p, alpha / np.sum(1.0 / np.arange(1, m + 1)))


def exceed(test_scores, cal_counts, thresholds, alpha, weights=None) -> np.ndarray:
    """EXCEED rejection mask for one test image."""
    return ebh(threshold_evalues(test_scores, cal_counts, thresholds, weights), alpha)


def single_threshold_rejects(n0: int, cal_total: int, n: int, alpha: float) -> bool:
    """Exact rejection event for one threshold (Prop. 4): all exceedances are flagged
    iff alpha (n + 1) N_0 >= N_0 + sum_i N_i (and N_0 > 0); otherwise nothing is."""
    return n0 > 0 and alpha * (n + 1) * n0 >= n0 + cal_total
