"""Post-registration robustness experiments on cached score maps.

Reuses the thresholds, decision rules, metrics and summaries of protocol.py unchanged:

- frontier: validity and power of each rule family over a range of its tuning parameter
  (max-nominal level, nominal percentile, EXCEED level), all judged against one target level;
  it also includes "FIX-max", the tuning-maximum threshold applied without EXCEED's count test,
  which isolates what the calibration step adds to the threshold choice;
- contamination: a fraction of the calibration images is replaced by defective images (which
  are then not test images), as in Corollary 1(ii).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage  # noqa: F401  (region_stats uses it through protocol)

from .baselines import max_nominal_threshold
from .evalues import exceedance_counts
from .metrics import dilate
from .protocol import (REGION_R, ImageCache, bh_tau, exceed_tau, region_stats, summarise,
                       threshold_grid)


def _rules_frontier():
    rules = {}
    for a in (0.05, 0.1, 0.2):
        rules[f"EX-max@{a}"] = ("ex", a)
        rules[f"EXR-max@{a}"] = ("exr", a)
    for a in (0.01, 0.02, 0.05, 0.1):
        rules[f"maxnom@{a}"] = ("maxnom", a)
    for q in (0.999, 0.9995, 0.9999, 0.99995):
        rules[f"pct{q}"] = ("pct", q)
    rules["pooledBY@0.1"] = ("by", 0.1)
    rules["pooledBH@0.1"] = ("bh", 0.1)
    rules["FIX-max"] = ("fix", "max")
    rules["FIX-q90"] = ("fix", "q90")
    return rules


def _rules_contam(alpha=0.1):
    return {f"EX-max@{alpha}": ("ex", alpha), f"EXR-max@{alpha}": ("exr", alpha),
            f"HYB-max@{alpha}": ("hyb", alpha), f"maxnom@{alpha}": ("maxnom", alpha),
            f"pooledBH@{alpha}": ("bh", alpha), f"pooledBY@{alpha}": ("by", alpha)}


def evaluate_rules(scores, masks, role, mode="frontier", frac=0.0, R=20, R_region=5, seed=0):
    rng = np.random.default_rng(seed)
    tune_idx = np.nonzero(role == "tune")[0]
    pool_idx = np.nonzero(role == "pool")[0]
    ano_idx = np.nonzero(role == "anomalous")[0]
    n_pool = len(pool_idx)
    n = n_pool - max(10, n_pool // 4)                  # as in the main protocol
    k_bad = min(int(round(frac * n)), max(len(ano_idx) - 10, 0)) if frac > 0 else 0
    S = scores
    m = S[0].size
    th = threshold_grid(S[tune_idx])
    keys = list(th)
    tvals = np.array([th[k] for k in keys])
    kmax, kq90 = keys.index("max"), keys.index("q90")
    counts = exceedance_counts(S, tvals)
    dils = [{r: dilate(masks[i], r) for r in REGION_R} for i in range(len(S))]
    comp = np.zeros((len(S), len(tvals), 1 + len(REGION_R)))
    for i in np.concatenate([ano_idx, pool_idx]):
        for k, tk in enumerate(tvals):
            comp[i, k] = region_stats(S[i] >= tk, masks[i], dils[i])
    maxes = S.reshape(len(S), -1).max(1)
    hm = np.sum(1.0 / np.arange(1, m + 1))
    rules = _rules_frontier() if mode == "frontier" else _rules_contam()
    rows = {name: [] for name in rules}
    cache = {}
    for r in range(R):
        perm = rng.permutation(pool_idx)
        if k_bad:
            bad = rng.permutation(ano_idx)
            cal = np.concatenate([perm[:n - k_bad], bad[:k_bad]])
            hold, test_ano = perm[n - k_bad:], bad[k_bad:]
        else:
            cal, hold, test_ano = perm[:n], perm[n:], ano_idx
        cal_sorted = np.sort(S[cal].ravel())
        N = cal_sorted.size
        cal_tot = counts[cal].sum(0)
        cal_ctot = comp[cal, :, 0].sum(0)
        # thresholds that depend only on the calibration draw: computed once per re-split
        pct = {par: float(np.quantile(cal_sorted, par)) for kind, par in rules.values() if kind == "pct"}
        fwer = {par: max_nominal_threshold(maxes[cal], par) for kind, par in rules.values() if kind == "maxnom"}
        for i in np.concatenate([test_ano, hold]):
            if i not in cache:
                cache[i] = ImageCache(S[i], masks[i])
            c = cache[i]
            is_anom = int(role[i] == "anomalous")
            n0 = counts[i]
            G = None
            for name, (kind, par) in rules.items():
                if kind == "ex":
                    tau = exceed_tau(n0[[kmax]], cal_tot[[kmax]], tvals[[kmax]], np.array([1.0]), n, m, par)
                elif kind == "exr":
                    c0, ctot = comp[i, kmax, 0], cal_ctot[kmax]
                    tau = tvals[kmax] if (c0 > 0 and par * (n + 1) * c0 >= c0 + ctot) else np.inf
                elif kind == "hyb":
                    t_ex = exceed_tau(n0[[kmax]], cal_tot[[kmax]], tvals[[kmax]], np.array([1.0]), n, m, par / 2)
                    tau = min(t_ex, max_nominal_threshold(maxes[cal], par / 2))
                elif kind == "maxnom":
                    tau = fwer[par]
                elif kind == "pct":
                    tau = pct[par]
                elif kind in ("bh", "by"):
                    if G is None:
                        G = N - np.searchsorted(cal_sorted, c.desc, side="left")
                    tau = bh_tau(c.desc, G, N, m, par if kind == "bh" else par / hm)
                elif kind == "fix":
                    tau = tvals[kmax] if par == "max" else tvals[kq90]
                else:
                    raise ValueError(kind)
                if np.isinf(tau):
                    reg = [0] + [0.0] * len(REGION_R)
                elif np.any(tvals == tau):
                    reg = list(comp[i, int(np.nonzero(tvals == tau)[0][0])])
                elif r < R_region:
                    reg = region_stats(S[i] >= tau, masks[i], dils[i])
                else:
                    reg = [np.nan] * (1 + len(REGION_R))
                rows[name].append([r, is_anom, i] + c.metrics(tau) + reg)
    out = {"mode": mode, "frac": frac, "k_bad": int(k_bad), "n_cal": int(n), "n_pool": int(n_pool),
           "n_anomalous": int(len(ano_idx)), "R": R, "R_region": R_region, "m": int(m),
           "thresholds": th,
           "summary": {name: summarise(np.asarray(rr, dtype=np.float64)) for name, rr in rows.items()}}
    return out
