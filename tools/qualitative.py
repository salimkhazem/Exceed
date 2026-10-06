"""Qualitative figure: flagged regions of EXCEED, MaxThr and pooled BH on real anomalous images.

One fixed calibration split per category (seeded). Selection rule, applied in a fixed category
order: the first anomalous images where EXCEED flags the defect with no false component and at
least one baseline flags a false component. The selection and every threshold are logged to
results/raw/qualitative.json."""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from exceed.baselines import max_nominal_threshold  # noqa: E402
from exceed.data import records  # noqa: E402
from exceed.evalues import exceedance_counts  # noqa: E402
from exceed.protocol import bh_tau, exceed_tau, load_npz  # noqa: E402

INK2 = "#52514e"
COL = {"EXCEED": "#2a78d6", "MaxThr": "#eda100", "pooled BH": "#eb6834"}


def false_components(R, A):
    lab, k = ndimage.label(R)
    hit = np.unique(lab[R & A])
    return k - int((hit > 0).sum()), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visa")
    ap.add_argument("--detector", default="patchcore")
    ap.add_argument("--categories", nargs="+", default=["candle", "pcb1", "cashew", "capsules"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default="paper/figures/fig_qual.pdf")
    a = ap.parse_args()
    picks, log = [], []
    for cat in a.categories:
        s, M, role, _ = load_npz(f"scores/{a.dataset}/{a.detector}/seed0/{cat}.npz")
        idx = np.load(f"scores/{a.dataset}/{a.detector}/seed0/{cat}.npz")["idx"]
        recs = records(a.dataset, cat)
        tune, pool, ano = (np.nonzero(role == r)[0] for r in ("tune", "pool", "anomalous"))
        rng = np.random.default_rng(0)
        perm = rng.permutation(pool)
        n = len(pool) - max(10, len(pool) // 4)
        cal = perm[:n]
        t = np.array([s[tune].max()])
        ctot = exceedance_counts(s[cal], t).sum(0)
        cs = np.sort(s[cal].ravel())
        mx = s[cal].reshape(n, -1).max(1)
        m = s[0].size
        for i in ano:
            n0 = exceedance_counts(s[i][None], t)[0]
            taus = {"EXCEED": exceed_tau(n0, ctot, t, np.array([1.0]), n, m, a.alpha),
                    "MaxThr": max_nominal_threshold(mx, a.alpha)}
            desc = np.sort(s[i].ravel())[::-1]
            G = cs.size - np.searchsorted(cs, desc, side="left")
            taus["pooled BH"] = bh_tau(desc, G, cs.size, m, a.alpha)
            Rs = {k: s[i] >= v for k, v in taus.items()}
            fc = {k: false_components(R, M[i]) for k, R in Rs.items()}
            ex_ok = (Rs["EXCEED"] & M[i]).any() and fc["EXCEED"][0] == 0
            base_bad = max(fc["MaxThr"][0], fc["pooled BH"][0]) >= 1
            if ex_ok and base_bad:
                picks.append((cat, i, recs[int(idx[i])][0], s[i], M[i], Rs))
                log.append({"category": cat, "row": int(i), "record": int(idx[i]), "n_cal": int(n),
                            "taus": {k: float(v) for k, v in taus.items()},
                            "false_components": {k: list(v) for k, v in fc.items()}})
                break
        if len(picks) >= a.k:
            break
    fig, axs = plt.subplots(len(picks), 5, figsize=(6.9, 1.45 * len(picks)), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for r, (cat, i, path, sm, A, Rs) in enumerate(picks):
        img = np.asarray(Image.open(path).convert("RGB").resize(A.shape[::-1]))
        axs[r, 0].imshow(img)
        axs[r, 0].contour(A, levels=[0.5], colors="white", linewidths=0.8)
        axs[r, 1].imshow(sm, cmap="Greys_r")
        for c, name in enumerate(["EXCEED", "MaxThr", "pooled BH"]):
            ax = axs[r, 2 + c]
            ax.imshow(img, alpha=0.55)
            ov = np.zeros(A.shape + (4,))
            rgb = matplotlib.colors.to_rgb(COL[name])
            ov[Rs[name]] = (*rgb, 0.85)
            ax.imshow(ov)
            ax.contour(A, levels=[0.5], colors="white", linewidths=0.6)
        for c in range(5):
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
        axs[r, 0].set_ylabel(cat, fontsize=7, color=INK2)
    for c, t in enumerate(["image, defect", "score map", "EXCEED", "MaxThr", "pooled BH"]):
        axs[0, c].set_title(t, fontsize=7)
    fig.savefig(a.out, dpi=200)
    json.dump({"selection_rule": __doc__, "picks": log}, open("results/raw/qualitative.json", "w"), indent=1)
    print(len(picks), "examples")


if __name__ == "__main__":
    main()
