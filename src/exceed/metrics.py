"""Per-image evaluation of a flagged mask R against a ground-truth defect mask A."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    st = ndimage.generate_binary_structure(2, 1)
    return ndimage.binary_dilation(mask.astype(bool), structure=st, iterations=r)


def image_metrics(R: np.ndarray, A: np.ndarray, buffer: int = 0) -> dict:
    """FDP (optionally excluding an r-pixel buffer around A from the nulls), TPR,
    defect-level detection (fraction of GT connected components hit by >= 1 flagged
    unit), and the number of flagged connected components that miss A entirely."""
    R = R.astype(bool)
    A = A.astype(bool)
    nR = int(R.sum())
    tp = int((R & A).sum())
    if buffer > 0:                         # FDP_r = |R cap H0^r| / max(|R|, 1)  (Prop. 5)
        v = int((R & ~dilate(A, buffer)).sum())
    else:
        v = nR - tp
    denom = max(nR, 1)
    out = {"n_flag": nR, "tp": tp, "fp": v, "fdp": v / denom,
           "defect_area": int(A.sum()), "any_flag": nR > 0}
    if A.any():
        out["tpr"] = tp / int(A.sum())
        lab, k = ndimage.label(A)
        hits = np.unique(lab[R & A])
        out["n_defects"] = int(k)
        out["n_defects_hit"] = int((hits > 0).sum())
    labR, kR = ndimage.label(R)
    if kR:
        overl = np.unique(labR[R & A])
        out["n_regions"] = int(kR)
        out["n_false_regions"] = int(kR - (overl > 0).sum())
    else:
        out["n_regions"] = 0
        out["n_false_regions"] = 0
    return out
