"""Evaluation protocol on cached score maps (paper Sec. 6).

Per (dataset, detector, fit-seed, category): the tuning split fixes the thresholds (and
the optional per-location standardisation); the remaining nominal pool is re-split R
times into n calibration images and held-out nominal test images; every rule is applied
to each test image (all anomalous images + held-out nominal images).

Every rule except per-location BH flags {u : S_0(u) >= tau} for an image-specific tau:
- EXCEED with a threshold mixture: its e-values are a step function of S_0, so e-BH
  selects one of the mixture thresholds (or nothing);
- BH/BY on pooled p-values: p is decreasing in S_0;
- fixed thresholds (max-nominal, percentiles, CRC, oracle).
Per-image metrics for threshold rules are therefore read off pre-sorted score arrays.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .baselines import crc_fpr_threshold, max_nominal_threshold, oracle_f1_threshold, perlocation_bh
from .evalues import exceedance_counts
from .metrics import dilate

ALPHAS = (0.05, 0.1, 0.2)
BUFFERS = (8, 16, 24, 32)
REGION_R = (0, 8)


def load_npz(path):
    z = np.load(path)
    s = z["scores"].astype(np.float32)
    masks = np.unpackbits(z["masks"], axis=-1)[..., : s.shape[-1]].astype(bool)
    return s, masks, z["role"], z["defect"]


def standardiser(tune: np.ndarray, kind: str):
    if kind == "raw":
        return lambda x: x
    if kind == "zloc":
        mu = tune.mean(0)
        sd = tune.std(0) + 1e-6
        return lambda x: (x - mu) / sd
    raise ValueError(kind)


def threshold_grid(tune_std: np.ndarray) -> dict:
    """EXCEED thresholds: a fixed function of the tuning split only (Assumption A1)."""
    mx = tune_std.reshape(len(tune_std), -1).max(1)
    q50, q90, qmax = np.quantile(mx, 0.5), np.quantile(mx, 0.9), mx.max()
    return {"q50": float(q50), "q90": float(q90), "max": float(qmax),
            "max+": float(qmax + 0.25 * (qmax - q50))}


EXCEED_VARIANTS = {             # name -> (threshold keys, weights)
    "EX-q90": (["q90"], [1.0]),
    "EX-max": (["max"], [1.0]),
    "EX-mix": (["q50", "q90", "max", "max+"], [0.25, 0.25, 0.25, 0.25]),
}


def exceed_tau(n0: np.ndarray, cal_tot: np.ndarray, tvals: np.ndarray, w: np.ndarray,
               n: int, m: int, alpha: float) -> float:
    """Exact e-BH decision for threshold-mixture EXCEED e-values on one image.

    n0[k] = N_0(t_k), cal_tot[k] = sum_i N_i(t_k). Returns the flagging threshold
    (units with S_0 >= tau are rejected) or +inf. Units with S_0 in [t_(j), t_(j+1))
    share the e-value L_j = sum_{i<=j} w_(i) c_(i), c = (n+1) m / (N_0 + sum_i N_i);
    e-BH rejects the block down to the lowest j with L_j >= m / (alpha N_0(t_(j)))."""
    order = np.argsort(tvals)
    t, n0s, ct, ws = tvals[order], n0[order], cal_tot[order], w[order]
    den = n0s + ct
    c = np.where((n0s > 0) & (den > 0), (n + 1) * m / np.maximum(den, 1), 0.0)
    level = np.cumsum(ws * c)            # e-value of units with S_0 in [t_(j), t_(j+1))
    tau = np.inf
    for j in range(len(t)):              # lowest threshold first = largest rejection set
        if n0s[j] > 0 and level[j] >= m / (alpha * n0s[j]):
            tau = t[j]
            break
    return float(tau)


def bh_tau(desc_scores: np.ndarray, G: np.ndarray, N: int, m: int, q: float) -> float:
    """BH on pooled p-values p = (1+G)/(N+1) with G = #{cal units >= s}: returns the
    score threshold (test scores sorted descending; G evaluated at those scores)."""
    k = np.arange(1, m + 1)
    ok = np.nonzero((1.0 + G) / (N + 1.0) <= q * k / m)[0]
    return float(desc_scores[ok.max()]) if ok.size else np.inf


class ImageCache:
    """Sorted score arrays for O(log m) metrics of any threshold rule."""

    def __init__(self, s0: np.ndarray, A: np.ndarray, buffers=BUFFERS):
        f = s0.ravel()
        a = A.ravel()
        self.sA = np.sort(f[a])
        self.sN = np.sort(f[~a])
        self.sNb = {b: np.sort(f[~dilate(A, b).ravel()]) for b in buffers}
        lab, k = ndimage.label(A)
        self.comp_max = (ndimage.maximum(s0, lab, index=np.arange(1, k + 1)) if k
                         else np.zeros(0))
        self.area = int(a.sum())
        self.desc = np.sort(f)[::-1]

    @staticmethod
    def _ge(arr, tau):
        return arr.size - np.searchsorted(arr, tau, side="left")

    def metrics(self, tau: float) -> list:
        tp = self._ge(self.sA, tau)
        fp = self._ge(self.sN, tau)
        nR = tp + fp
        row = [fp / max(nR, 1), tp / self.area if self.area else np.nan, float(nR > 0),
               int((self.comp_max >= tau).sum()), len(self.comp_max), tp, fp, self.area]
        for b, arr in self.sNb.items():          # FDP_r = |R \cap H0^r| / max(|R|, 1)
            fpb = self._ge(arr, tau)
            row.append(fpb / max(nR, 1))
        return row

    def metrics_mask(self, R: np.ndarray, A: np.ndarray, s0: np.ndarray) -> list:
        tp = int((R & A).sum())
        fp = int(R.sum()) - tp
        lab, k = ndimage.label(A)
        hit = len(np.unique(lab[R & A])) - (1 if (R & A).any() and 0 in lab[R & A] else 0)
        row = [fp / max(tp + fp, 1), tp / self.area if self.area else np.nan, float(tp + fp > 0),
               hit, k, tp, fp, self.area]
        for b in self.sNb:
            fpb = int((R & ~dilate(A, b)).sum())
            row.append(fpb / max(tp + fp, 1))
        return row


COLS = (["resplit", "is_anom", "img", "fdp", "tpr", "any_flag", "defects_hit", "defects", "tp",
         "fp", "area"] + [f"fdp_b{b}" for b in BUFFERS]
        + ["n_regions"] + [f"rfdp_r{r}" for r in REGION_R])


def region_stats(R: np.ndarray, A: np.ndarray, dil: dict) -> list:
    """Number of flagged connected components and, for each r in REGION_R, the fraction
    of them that do not intersect the r-dilated defect (region-level FDP)."""
    lab, k = ndimage.label(R)
    if k == 0:
        return [0] + [0.0] * len(REGION_R)
    out = [k]
    for r in REGION_R:
        hit = np.unique(lab[R & dil[r]])
        out.append((k - int((hit > 0).sum())) / k)
    return out


def evaluate_category(scores, masks, role, alphas=ALPHAS, n_cal=None, R=100, R_perloc=10,
                      kinds=("raw",), seed=0, crc_gamma=1e-3, R_region=5):
    rng = np.random.default_rng(seed)
    tune_idx = np.nonzero(role == "tune")[0]
    pool_idx = np.nonzero(role == "pool")[0]
    ano_idx = np.nonzero(role == "anomalous")[0]
    n_pool = len(pool_idx)
    n = n_cal if n_cal is not None else n_pool - max(10, n_pool // 4)
    if n < 10 or n_pool - n < 5:
        raise ValueError(f"pool too small: n_pool={n_pool}, n={n}")
    m = scores[0].size
    out = {"n_cal": int(n), "n_pool": int(n_pool), "n_anomalous": int(len(ano_idx)),
           "n_tune": int(len(tune_idx)), "R": R, "R_perloc": R_perloc, "m": int(m),
           "rules": {}, "thresholds": {}}
    for kind in kinds:
        f = standardiser(scores[tune_idx], kind)
        S = f(scores)
        th = threshold_grid(S[tune_idx])
        # design rule (tuning split only): smallest threshold whose mean number of
        # connected components per tuning image is <= alpha / 2
        mx = S[tune_idx].reshape(len(tune_idx), -1).max(1)
        cand = np.unique(np.quantile(mx, np.linspace(0.0, 1.0, 41)))
        rate = np.array([np.mean([ndimage.label(S[j] >= t)[1] for j in tune_idx]) for t in cand])
        for a in alphas:
            ok = np.nonzero(rate <= a / 2)[0]
            th[f"design{a}"] = float(cand[ok.min()]) if ok.size else th["max+"]
        keys = list(th)
        tvals = np.array([th[k] for k in keys])
        counts = exceedance_counts(S, tvals)                         # (N, K), once
        dils = [{r: dilate(masks[i], r) for r in REGION_R} for i in range(len(S))]
        comp = np.zeros((len(S), len(tvals), 1 + len(REGION_R)))
        for i in np.concatenate([ano_idx, pool_idx]):
            for k, tk in enumerate(tvals):
                comp[i, k] = region_stats(S[i] >= tk, masks[i], dils[i])
        grid = np.unique(np.quantile(S[tune_idx], 1 - np.logspace(-7, -1, 300)))
        gcounts = exceedance_counts(S, grid)                         # CRC grid, once
        maxes = S.reshape(len(S), -1).max(1)
        ora = oracle_f1_threshold(S[ano_idx], masks[ano_idx],
                                  np.quantile(S[ano_idx], np.linspace(0.5, 0.99995, 400)))
        out["thresholds"][kind] = th | {"oracle_f1": ora}
        cache = {}
        rows = {}
        diag = []
        for r in range(R):
            perm = rng.permutation(pool_idx)
            cal, hold = perm[:n], perm[n:]
            cal_sorted = np.sort(S[cal].ravel())
            N = cal_sorted.size
            cal_tot = counts[cal].sum(0)
            cal_ctot = comp[cal, :, 0].sum(0)
            fixed = {"pct99.9": float(np.quantile(cal_sorted, 0.999)),
                     "pct99.99": float(np.quantile(cal_sorted, 0.9999)),
                     "crcFPR1e-3": crc_fpr_threshold(gcounts[cal], grid, m, crc_gamma),
                     "oracleF1": ora}
            fwer = {a: max_nominal_threshold(maxes[cal], a) for a in alphas}
            for i in np.concatenate([ano_idx, hold]):
                if i not in cache:
                    cache[i] = ImageCache(S[i], masks[i])
                c = cache[i]
                is_anom = int(role[i] == "anomalous")
                n0 = counts[i]
                G = N - np.searchsorted(cal_sorted, c.desc, side="left")
                hm = np.sum(1.0 / np.arange(1, m + 1))
                taus = {}
                for a in alphas:
                    for name, (ks, w) in EXCEED_VARIANTS.items():
                        sel = [keys.index(k) for k in ks]
                        taus[(name, a)] = exceed_tau(n0[sel], cal_tot[sel], tvals[sel],
                                                     np.array(w), n, m, a)
                    kd = keys.index(f"design{a}")
                    taus[("EX-design", a)] = exceed_tau(n0[[kd]], cal_tot[[kd]], tvals[[kd]],
                                                        np.array([1.0]), n, m, a)
                    # hybrid (valid): EXCEED-max at a/2 OR max-nominal at a/2 (FDP of a union
                    # is at most the sum of FDPs); union of superlevel sets = lower threshold
                    kmax = keys.index("max")
                    t_ex = exceed_tau(n0[[kmax]], cal_tot[[kmax]], tvals[[kmax]], np.array([1.0]),
                                      n, m, a / 2)
                    taus[("HYB-max", a)] = min(t_ex, max_nominal_threshold(maxes[cal], a / 2))
                    # two-stage heuristic (not FDR-valid): conformal image-level test on the
                    # image maximum at level a, then flag the superlevel set of the q90 threshold
                    p_img = (1 + int((maxes[cal] >= maxes[i]).sum())) / (n + 1)
                    taus[("2stage", a)] = tvals[keys.index("q90")] if p_img <= a else np.inf
                    taus[("pooledBH", a)] = bh_tau(c.desc, G, N, m, a)
                    taus[("pooledBY", a)] = bh_tau(c.desc, G, N, m, a / hm)
                    taus[("maxnom", a)] = fwer[a]
                for name, t in fixed.items():
                    taus[(name, None)] = t
                for a in alphas:                     # region-level EXCEED, one threshold each
                    for key in ("q90", "max", "max+", f"design{a}"):
                        k = keys.index(key)
                        c0, ctot = comp[i, k, 0], cal_ctot[k]
                        ok = c0 > 0 and a * (n + 1) * c0 >= c0 + ctot
                        taus[("EXR-design" if key.startswith("design") else f"EXR-{key}", a)] = \
                            tvals[k] if ok else np.inf
                # dominance diagnostic (Thm 1(b)): null-part image e-value of single-level
                # EXCEED at the 'max' threshold, sum_{u in H0} e(u) / m; its mean must be <= 1
                km = keys.index("max")
                den = n0[km] + cal_tot[km]
                v0 = ImageCache._ge(c.sN, tvals[km])
                v24 = ImageCache._ge(c.sNb[24], tvals[km])
                diag.append([r, is_anom, (n + 1) * v0 / den if den else 0.0,
                             (n + 1) * v24 / den if den else 0.0])
                for (name, a), tau in taus.items():
                    if np.isinf(tau):
                        reg = [0] + [0.0] * len(REGION_R)
                    elif np.any(tvals == tau):
                        reg = list(comp[i, int(np.nonzero(tvals == tau)[0][0])])
                    elif r < R_region:
                        reg = region_stats(S[i] >= tau, masks[i], dils[i])
                    else:
                        reg = [np.nan] * (1 + len(REGION_R))
                    rows.setdefault((f"{name}/{kind}", a), []).append(
                        [r, is_anom, i] + c.metrics(tau) + reg)
                if r < R_perloc:
                    for a in alphas:
                        Rm = perlocation_bh(S[i], S[cal], a)
                        rows.setdefault((f"perlocBH/{kind}", a), []).append(
                            [r, is_anom, i] + c.metrics_mask(Rm, masks[i], S[i])
                            + region_stats(Rm, masks[i], dils[i]))
        for (name, a), rr in rows.items():
            out["rules"][f"{name}@{a}"] = np.asarray(rr, dtype=np.float64)
        dg = np.asarray(diag, dtype=np.float64)
        an = dg[:, 1] == 1
        out["dominance_diag"] = {"null_eimg_mean_r0": float(dg[an, 2].mean()),
                                 "null_eimg_se_r0": float(dg[an, 2].std() / np.sqrt(an.sum())),
                                 "null_eimg_mean_r24": float(dg[an, 3].mean()),
                                 "null_eimg_se_r24": float(dg[an, 3].std() / np.sqrt(an.sum())),
                                 "nominal_eimg_mean": float(dg[~an, 2].mean())}
    return out


def summarise(arr: np.ndarray) -> dict:
    """Per-image averages over calibration re-splits, then means and standard errors over
    test images (the finite set of anomalous images is the main source of uncertainty)."""
    d = {c: arr[:, i] for i, c in enumerate(COLS)}
    an = d["is_anom"] == 1
    no = ~an
    out = {}

    def img_mean(col, mask):
        ids = np.unique(d["img"][mask])
        v = np.array([np.nanmean(d[col][mask & (d["img"] == j)]) for j in ids])
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan"), float("nan"), 0
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0, int(v.size)

    for name, col in [("fdr_anom", "fdp")] + [(f"fdr_anom_b{b}", f"fdp_b{b}") for b in BUFFERS] + \
            [(f"fdr_region_r{r}", f"rfdp_r{r}") for r in REGION_R]:
        mu, se, k = img_mean(col, an)
        out[name], out[name + "_se"] = mu, se
    out["n_anom_images"] = k
    # nominal false alarms: held-out nominal images differ across re-splits -> per re-split means
    ps = [d["any_flag"][no & (d["resplit"] == r)].mean() for r in np.unique(d["resplit"])
          if (no & (d["resplit"] == r)).any()]
    out["fa_nominal"] = float(np.mean(ps)) if ps else float("nan")
    fa_all = d["any_flag"][no]
    out["fa_nominal_se"] = float(fa_all.std(ddof=1) / np.sqrt(max(len(np.unique(d["img"][no])), 1))) if fa_all.size > 1 else 0.0
    out["n_nominal_rows"] = int(no.sum())
    tp, fp, area = d["tp"][an].sum(), d["fp"][an].sum(), d["area"][an].sum()
    out["tpr_anom_mean"] = float(np.nanmean(d["tpr"][an]))
    out["pixel_tpr_pooled"] = float(tp / max(area, 1))
    out["defect_detection"] = float(d["defects_hit"][an].sum() / max(d["defects"][an].sum(), 1))
    out["image_detection"] = float((d["tp"][an] > 0).mean())
    out["f1_anom_pooled"] = float(2 * tp / max(2 * tp + fp + (area - tp), 1))
    out["n_resplits"] = int(len(np.unique(d["resplit"])))
    return out
