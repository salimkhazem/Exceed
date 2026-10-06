"""Fit a frozen-feature scorer on the fit split and cache score maps for every
non-fit image: scores/<dataset>/<detector>/seed<k>/<category>.npz (+ .json record)."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from exceed.data import MVTEC_CATEGORIES, VISA_CATEGORIES, fingerprint, fit_tune_split, records  # noqa: E402
from exceed.detectors import DETECTORS, load_mask  # noqa: E402
from exceed.provenance import Timer, atomic_json, run_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mvtec", "visa"], required=True)
    ap.add_argument("--detector", choices=list(DETECTORS), required=True)
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="scores")
    a = ap.parse_args()
    cats = a.categories or (MVTEC_CATEGORIES if a.dataset == "mvtec" else VISA_CATEGORIES)
    torch.manual_seed(a.seed)
    for cat in cats:
        path = os.path.join(a.out, a.dataset, a.detector, f"seed{a.seed}", f"{cat}.npz")
        if os.path.exists(path):
            print("skip", path)
            continue
        recs = records(a.dataset, cat)
        nom = [i for i, r in enumerate(recs) if not r[2]]
        ano = [i for i, r in enumerate(recs) if r[2]]
        fit_i, tune_i, rest_i = fit_tune_split(len(nom), a.seed)
        fit = [recs[nom[j]][0] for j in fit_i]
        keep = [nom[j] for j in tune_i] + [nom[j] for j in rest_i] + ano
        role = (["tune"] * len(tune_i)) + (["pool"] * len(rest_i)) + (["anomalous"] * len(ano))
        det = DETECTORS[a.detector](device="cuda", seed=a.seed)
        with Timer() as tm:
            det.fit(fit)
            maps = det.score([recs[i][0] for i in keep])
        masks = np.stack([load_mask(recs[i][1], det.size) for i in keep])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, scores=maps.astype(np.float32), masks=np.packbits(masks, axis=-1),
                            role=np.array(role), idx=np.array(keep),
                            defect=np.array([recs[i][3] for i in keep]))
        rec = run_record(vars(a) | {"category": cat})
        rec.update({"n_fit": len(fit), "n_tune": len(tune_i), "n_pool": len(rest_i),
                    "n_anomalous": len(ano), "records_sha256": fingerprint(recs),
                    "seconds": tm.seconds, "peak_gpu_bytes": tm.peak_gpu_bytes,
                    "map_size": det.size, "score_dtype": "float32"})
        atomic_json(rec, path.replace(".npz", ".json"))
        print(cat, json.dumps({k: rec[k] for k in ("n_fit", "n_tune", "n_pool", "n_anomalous",
                                                   "seconds", "peak_gpu_bytes")}), flush=True)
        del det
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
