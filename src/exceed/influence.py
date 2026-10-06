"""Influence radius of a fitted scorer, measured on TUNING images only (no calibration or
test data): paste a square patch from another tuning image, re-score, and record how far
the score increase spreads from the pasted region (paper Sec. 5, Prop. 5)."""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from .detectors import MEAN, STD, load_image

BANDS = (0, 4, 8, 16, 24, 32, 48, 64, 10_000)


@torch.no_grad()
def influence_profile(det, tune_paths, seed=0, n_images=16, sizes=(8, 16, 32)):
    rng = np.random.default_rng(seed)
    paths = list(tune_paths)[: max(n_images, 2)]
    S, Sin = det.size, det.input_size
    out = []
    for i, p in enumerate(paths[:n_images]):
        base = load_image(p, Sin)
        donor = load_image(paths[(i + 1) % len(paths)], Sin)
        for s in sizes:
            k = int(round(s * Sin / S))
            y, x = rng.integers(0, Sin - k, 2)
            dy, dx = rng.integers(0, Sin - k, 2)
            pert = base.clone()
            pert[:, y:y + k, x:x + k] = donor[:, dy:dy + k, dx:dx + k]
            xb = torch.stack([base, pert])
            maps = det.score_tensor(((xb - MEAN) / STD).to(det.device))
            A = np.zeros((S, S), bool)
            ys, xs = int(y * S / Sin), int(x * S / Sin)
            A[ys:ys + s, xs:xs + s] = True
            d = ndimage.distance_transform_edt(~A)
            diff = maps[1] - maps[0]
            band_max = []
            for lo, hi in zip(BANDS[:-1], BANDS[1:]):
                sel = (d > lo) & (d <= hi) if lo > 0 else (d <= hi)
                band_max.append(float(diff[sel].max()) if sel.any() else 0.0)
            out.append({"image": i, "size": s, "band_max_increase": band_max,
                        "inside_max_increase": float(diff[A].max()),
                        "base_max": float(maps[0].max()), "base_median": float(np.median(maps[0]))})
    return {"bands": list(BANDS), "insertions": out}
