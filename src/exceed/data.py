"""MVTec AD and VisA loaders (original directory layouts) and split manifests.

Every record is (image_path, mask_path or None, is_anomalous, defect_type, source_split).
Nominal images from the official train and test splits are pooled (the protocol
re-splits them into fit / tune / calibration / held-out nominal test); anomalous
images are always test images.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os

import numpy as np

# Dataset root: set EXCEED_DATA to the directory holding mvtec_ad/ and visa/VisA/ (see README).
ROOT = os.environ.get("EXCEED_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MVTEC = os.path.join(ROOT, "mvtec_ad")
VISA = os.path.join(ROOT, "visa", "VisA")

MVTEC_CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
                    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
                    "zipper"]
VISA_CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
                   "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def _img_files(d):
    out = []
    for ext in ("png", "PNG", "jpg", "JPG", "jpeg", "bmp"):
        out += glob.glob(os.path.join(d, f"*.{ext}"))
    return sorted(out)


def mvtec_records(category: str, root: str = MVTEC):
    base = os.path.join(root, category)
    recs = []
    for p in _img_files(os.path.join(base, "train", "good")):
        recs.append((p, None, False, "good", "train"))
    for d in sorted(os.listdir(os.path.join(base, "test"))):
        for p in _img_files(os.path.join(base, "test", d)):
            if d == "good":
                recs.append((p, None, False, "good", "test"))
            else:
                stem = os.path.splitext(os.path.basename(p))[0]
                mp = os.path.join(base, "ground_truth", d, f"{stem}_mask.png")
                if not os.path.exists(mp):
                    raise FileNotFoundError(mp)
                recs.append((p, mp, True, d, "test"))
    return recs


def visa_records(category: str, root: str = VISA):
    """VisA with the official one-class split file split_csv/1cls.csv."""
    split = os.path.join(root, "split_csv", "1cls.csv")
    recs = []
    with open(split) as fh:
        for row in csv.DictReader(fh):
            if row["object"] != category:
                continue
            p = os.path.join(root, row["image"])
            anom = row["label"].strip().lower() == "anomaly"
            mp = os.path.join(root, row["mask"]) if anom and row.get("mask") else None
            recs.append((p, mp, anom, row["label"].strip().lower(), row["split"].strip()))
    if not recs:
        raise ValueError(f"no VisA rows for {category}")
    return recs


def records(dataset: str, category: str):
    return mvtec_records(category) if dataset == "mvtec" else visa_records(category)


def fingerprint(recs) -> str:
    h = hashlib.sha256()
    for r in recs:
        h.update(repr(r[:3]).encode())
    return h.hexdigest()


def fit_tune_split(n_nominal: int, seed: int, fit_frac: float = 0.4, fit_max: int = 200,
                   tune_frac: float = 0.1, tune_min: int = 8):
    """Indices (into the nominal pool) for the scorer fit split and the tuning split.
    The remainder is the calibration / held-out-nominal pool, re-split later on CPU."""
    rng = np.random.default_rng(10_000 + seed)
    perm = rng.permutation(n_nominal)
    n_fit = min(fit_max, int(round(fit_frac * n_nominal)))
    n_tune = max(tune_min, int(round(tune_frac * n_nominal)))
    if n_fit + n_tune >= n_nominal - 10:
        raise ValueError("nominal pool too small")
    return perm[:n_fit], perm[n_fit:n_fit + n_tune], perm[n_fit + n_tune:]


def write_manifest(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s = json.dumps(payload, sort_keys=True, indent=1)
    with open(path, "w") as fh:
        fh.write(s)
    return hashlib.sha256(s.encode()).hexdigest()
