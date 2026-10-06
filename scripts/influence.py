"""Refit a scorer on the same fit split as scripts/score.py and measure its influence
radius on tuning images only. Writes scores/<ds>/<det>/seed<k>/<cat>.influence.json."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from exceed.data import MVTEC_CATEGORIES, VISA_CATEGORIES, fit_tune_split, records  # noqa: E402
from exceed.detectors import DETECTORS  # noqa: E402
from exceed.influence import influence_profile  # noqa: E402
from exceed.provenance import atomic_json, run_record  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
ap.add_argument("--detector", choices=list(DETECTORS), required=True)
ap.add_argument("--categories", nargs="*")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
for cat in a.categories or (MVTEC_CATEGORIES if a.dataset == "mvtec" else VISA_CATEGORIES):
    out = os.path.join("scores", a.dataset, a.detector, f"seed{a.seed}", f"{cat}.influence.json")
    if os.path.exists(out):
        continue
    recs = records(a.dataset, cat)
    nom = [i for i, r in enumerate(recs) if not r[2]]
    fit_i, tune_i, _ = fit_tune_split(len(nom), a.seed)
    torch.manual_seed(a.seed)
    det = DETECTORS[a.detector](device="cuda", seed=a.seed)
    det.fit([recs[nom[j]][0] for j in fit_i])
    rec = run_record(vars(a) | {"category": cat})
    rec["influence"] = influence_profile(det, [recs[nom[j]][0] for j in tune_i], seed=a.seed)
    atomic_json(rec, out)
    print(cat, "done", flush=True)
    del det
    torch.cuda.empty_cache()
