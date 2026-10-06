"""Calibration-size sweep (post-registration robustness experiment).

Runs the unchanged evaluation job of scripts/evaluate.py with a fixed number n of calibration
images, on every category whose nominal pool allows it (n >= 10 and at least 5 held-out nominal
images). Writes results/raw/nsweep<n>/<dataset>/<detector>/seed<k>/<category>.{npz,json}.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import job  # noqa: E402  (the per-category job of the main evaluation)


def pool_size(npz_path):
    return json.load(open(npz_path[:-4] + ".json"))["n_pool"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument("--glob", default="scores/*/*/seed0/*.npz")
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--R-perloc", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.glob))
    jobs = []
    for n in a.n:
        ok = [p for p in paths if n >= 10 and pool_size(p) - n >= 5]
        print(f"n={n}: {len(ok)} of {len(paths)} categories", flush=True)
        jobs += [(p, f"nsweep{n}", a.R, a.R_perloc, n, ("raw",)) for p in ok]
    with Pool(a.workers) as pool:
        for out, msg in pool.imap_unordered(job, jobs):
            print(out, msg, flush=True)


if __name__ == "__main__":
    main()
