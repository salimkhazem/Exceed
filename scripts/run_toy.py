"""Synthetic experiments with known ground truth (paper Sec. 6.1; Figs. 1-2).

T1  validity/power on Gaussian fields with location heterogeneity (+ image effects)
T2  pooled-p BH on the intensity-extent law of Prop. 3 as n grows
T3  sharp localisation boundary (Thm. 2): detection and FDR vs defect exceedances a/lambda
T4  resolution barrier of per-location conformal p-values vs defect area
Writes results/raw/toy/<exp>.json with a provenance record."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from exceed.baselines import max_nominal_threshold, perlocation_bh  # noqa: E402
from exceed.evalues import bh, by, exceedance_counts  # noqa: E402
from exceed.protocol import exceed_tau  # noqa: E402
from exceed.provenance import atomic_json, run_record  # noqa: E402
from exceed.theory import barrier_min_rejections  # noqa: E402
from exceed.toy import ToyFields, hot_pixel_law  # noqa: E402


def fdp_tpr(R, A):
    v = int((R & ~A).sum())
    t = int((R & A).sum())
    return v / max(int(R.sum()), 1), (t / int(A.sum()) if A.any() else np.nan)


def t1(reps, seed):
    """Validity/power on heterogeneous fields: scores standardised per location on the tuning
    split; EXCEED thresholds target lambda nominal exceedances per image (tuning split)."""
    out = []
    for amp in (3.0, 4.0, 5.0, 6.0):
        for gsd in (0.0, 0.5):
            toy = ToyFields(size=64, seed=seed + int(10 * amp) + int(100 * gsd), global_sd=gsd)
            tune = np.stack([toy.field() for _ in range(200)])
            mu, sd = tune.mean(0), tune.std(0) + 1e-6
            z = lambda x: (x - mu) / sd
            tz = z(tune)
            n, m, alpha = 100, 64 * 64, 0.1
            lam = {"EX(l=0.5)": 0.5, "EX(l=2)": 2.0}
            th = {k: float(np.quantile(tz, 1 - v / m)) for k, v in lam.items()}
            rows = {k: [] for k in list(lam) + ["EX-avg", "pooledBH", "pooledBY", "perlocBH",
                                                  "MaxThr", "HYB"]}
            tv = np.array(list(th.values()))
            for r in range(reps):
                cal = z(np.stack([toy.field() for _ in range(n)]))
                ctot = exceedance_counts(cal, tv).sum(0)
                if r % 2:
                    s, A = toy.field(), np.zeros((64, 64), bool)
                else:
                    s, A = toy.anomalous(amp)
                s = z(s)
                n0 = exceedance_counts(s[None], tv)[0]
                Rs = {}
                for j, k in enumerate(lam):
                    Rs[k] = s >= exceed_tau(n0[[j]], ctot[[j]], tv[[j]], np.array([1.0]), n, m, alpha)
                Rs["EX-avg"] = s >= exceed_tau(n0, ctot, tv, np.array([0.5, 0.5]), n, m, alpha)
                cs = np.sort(cal.ravel())
                p = ((1 + cs.size - np.searchsorted(cs, s.ravel(), side="left")) / (cs.size + 1)).reshape(s.shape)
                mxc = cal.reshape(n, -1).max(1)
                t_half = exceed_tau(n0[[0]], ctot[[0]], tv[[0]], np.array([1.0]), n, m, alpha / 2)
                Rs.update({"pooledBH": bh(p, alpha), "pooledBY": by(p, alpha),
                           "perlocBH": perlocation_bh(s, cal, alpha),
                           "MaxThr": s >= max_nominal_threshold(mxc, alpha),
                           "HYB": s >= min(t_half, max_nominal_threshold(mxc, alpha / 2))})
                for k, R in Rs.items():
                    rows[k].append((r % 2 == 1,) + fdp_tpr(R, A))
            for k, v in rows.items():
                v = np.array(v, dtype=float)
                an = v[:, 0] == 0
                out.append({"amp": amp, "global_sd": gsd, "rule": k,
                            "fdr_anom": float(v[an, 1].mean()),
                            "fdr_anom_se": float(v[an, 1].std() / np.sqrt(an.sum())),
                            "fa_nominal": float(v[~an, 1].mean()),
                            "tpr": float(np.nanmean(v[an, 2]))})
    return out


def t2(reps, seed):
    rng = np.random.default_rng(seed)
    m, alpha = 65536, 0.1
    lmax = 0.95 * np.log(m) * alpha / 2
    out = []
    for n in (20, 50, 100, 500, 2000, 5000):
        bh_rej = by_rej = ex_rej = 0
        H = np.sum(1.0 / np.arange(1, m + 1))
        for _ in range(reps):
            Ls, Ks = zip(*[hot_pixel_law(m, alpha, rng) for _ in range(n)])
            Ls, Ks = np.array(Ls), np.array(Ks)
            L0, K0 = hot_pixel_law(m, alpha, rng)
            p_hot = (1 + Ks[Ls >= L0].sum()) / (n * m + 1)
            bh_rej += p_hot <= alpha * K0 / m
            by_rej += p_hot <= alpha * K0 / (m * H)
            tL = 0.25 * lmax
            n0 = K0 * (L0 >= tL)
            ex_rej += bool(n0 > 0 and alpha * (n + 1) * n0 >= n0 + Ks[Ls >= tL].sum())
        out.append({"n": n, "pooledBH_fdr": bh_rej / reps, "pooledBY_fdr": by_rej / reps,
                    "EXCEED_fdr": ex_rej / reps, "reps": reps})
    return out


def t3(reps, seed):
    rng = np.random.default_rng(seed)
    alpha, lam = 0.1, 20
    out = []
    for n in (20, 50, 100, 400):
        for ratio in np.linspace(0, 20, 41):
            a = int(round(ratio * lam))
            det = fdp = 0.0
            for _ in range(reps):
                cal_counts = rng.poisson(lam, n)             # nominal exceedance counts
                V = rng.poisson(lam)                          # null exceedances in the test image
                N0 = V + a
                ok = N0 > 0 and alpha * (n + 1) * N0 >= N0 + cal_counts.sum()
                det += ok and a > 0
                fdp += (V / N0) if ok else 0.0
            thr_fixed = lam * (1 - alpha) * (n + 1) / (alpha * (n + 1) - 1)
            out.append({"n": n, "a_over_lambda": a / lam, "detect": det / reps, "fdr": fdp / reps,
                        "upper_bound": min(1.0, alpha * (lam + a) / lam),
                        "exceed_threshold_a": thr_fixed, "limit_threshold_a": lam * (1 - alpha) / alpha})
    return out


def t4(reps, seed):
    rng = np.random.default_rng(seed)
    H, alpha, n = 64, 0.1, 50
    m = H * H
    kmin = barrier_min_rejections(m, n, alpha)
    out = []
    for side in (2, 4, 8, 12, 16, 24, 32):
        d_pl = d_ex = 0
        for _ in range(reps):
            cal = rng.standard_normal((n, H, H))
            s = rng.standard_normal((H, H))
            A = np.zeros((H, H), bool)
            y, x = rng.integers(0, H - side, 2)
            A[y:y + side, x:x + side] = True
            s[A] += 8.0
            d_pl += bool(perlocation_bh(s, cal, alpha)[A].any())
            t = np.array([4.0])
            n0 = exceedance_counts(s[None], t)[0]
            tau = exceed_tau(n0, exceedance_counts(cal, t).sum(0), t, np.array([1.0]), n, m, alpha)
            d_ex += bool((s >= tau)[A].any())
        out.append({"side": side, "area": side * side, "perlocBH_detect": d_pl / reps,
                    "EXCEED_detect": d_ex / reps, "barrier_min_rejections": kmin})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["t1", "t2", "t3", "t4"])
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for e in a.exp:
        res = {"t1": t1, "t2": t2, "t3": t3, "t4": t4}[e](a.reps, a.seed)
        rec = run_record(vars(a) | {"exp": e})
        rec["results"] = res
        atomic_json(rec, f"results/raw/toy/{e}.json")
        print(e, "done", flush=True)
