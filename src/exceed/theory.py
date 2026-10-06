"""Closed-form predictions used to check the theory against experiments."""
from __future__ import annotations

import math


def barrier_min_rejections(m: int, n: int, alpha: float) -> int:
    """Prop. 2: with per-location conformal p-values (>= 1/(n+1)), BH can only reject
    sets of at least ceil(m / (alpha (n + 1))) units."""
    return math.ceil(m / (alpha * (n + 1)))


def detection_threshold(cal_total: int, n: int, alpha: float) -> float:
    """Prop. 4 (single threshold): exceedances are flagged iff
    N_0 >= sum_i N_i / (alpha (n + 1) - 1). Returns that critical N_0 (inf if
    alpha (n + 1) <= 1, i.e. no rejection is possible, Cor. 1(iii))."""
    c = alpha * (n + 1) - 1.0
    return math.inf if c <= 0 else cal_total / c


def asymptotic_detection_threshold(lam: float, alpha: float) -> float:
    """Large-n limit of `detection_threshold`: lambda(t) / alpha."""
    return lam / alpha


def prop3_rejects_all(m: int, alpha: float) -> bool:
    """Prop. 3 limit: pooled-p BH rejects the hot units of every nominal image of the
    intensity-extent law as n -> infinity whenever ln m > 1/alpha."""
    return math.log(m) > 1.0 / alpha
