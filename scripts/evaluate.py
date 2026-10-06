"""Run the evaluation protocol on cached score maps (CPU, one process per category).

Writes results/raw/<tag>/<dataset>/<detector>/seed<k>/<category>.npz (per-image rows for
every rule) and .json (summaries + provenance)."""
import argparse
import glob
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from exceed.protocol import evaluate_category, load_npz, summarise  # noqa: E402
from exceed.provenance import Timer, atomic_json, run_record, sha256_file  # noqa: E402


def job(args):
    path, tag, R, R_perloc, n_cal, kinds = args
    parts = path.split(os.sep)
    ds, det, seed, cat = parts[-4], parts[-3], parts[-2], parts[-1][:-4]
    out = os.path.join("results", "raw", tag, ds, det, seed, cat)
    if os.path.exists(out + ".json"):
        return out, "skip"
    s, masks, role, _ = load_npz(path)
    with Timer() as tm:
        res = evaluate_category(s, masks, role, R=R, R_perloc=R_perloc, n_cal=n_cal,
                                kinds=kinds, seed=int(hashlib.sha256(f"{ds}|{det}|{seed}|{cat}".encode()).hexdigest(), 16) % 2**31)
    rules = res.pop("rules")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + ".npz", **{k.replace("/", "|"): v for k, v in rules.items()})
    rec = run_record({"tag": tag, "scores": path, "R": R, "R_perloc": R_perloc, "n_cal": n_cal,
                      "kinds": list(kinds)})
    rec.update(res)
    rec["scores_sha256"] = sha256_file(path)
    rec["seconds"] = tm.seconds
    rec["summary"] = {k: summarise(v) for k, v in rules.items()}
    atomic_json(rec, out + ".json")
    return out, f"{tm.seconds:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--glob", default="scores/*/*/seed*/*.npz")
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--R-perloc", type=int, default=10)
    ap.add_argument("--n-cal", type=int, default=None)
    ap.add_argument("--kinds", nargs="+", default=["raw"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.glob))
    jobs = [(p, a.tag, a.R, a.R_perloc, a.n_cal, tuple(a.kinds)) for p in paths]
    with Pool(a.workers) as pool:
        for out, msg in pool.imap_unordered(job, jobs):
            print(out, msg, flush=True)


if __name__ == "__main__":
    main()
