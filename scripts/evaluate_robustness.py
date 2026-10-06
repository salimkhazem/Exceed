"""Post-registration robustness experiments (see src/exceed/robustness.py).

  python scripts/evaluate_robustness.py --mode frontier
  python scripts/evaluate_robustness.py --mode contam --frac 0.05 0.1 0.2

Writes results/raw/<tag>/<dataset>/<detector>/seed<k>/<category>.json with tag "frontier" or
"contam<percent>"."""
import argparse
import glob
import hashlib
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from exceed.protocol import load_npz  # noqa: E402
from exceed.provenance import Timer, atomic_json, run_record, sha256_file  # noqa: E402
from exceed.robustness import evaluate_rules  # noqa: E402


def job(args):
    path, mode, frac, R = args
    ds, det, seed, cat = path.split(os.sep)[-4:]
    cat = cat[:-4]
    tag = "frontier" if mode == "frontier" else f"contam{int(round(100 * frac)):02d}"
    out = os.path.join("results", "raw", tag, ds, det, seed, cat)
    if os.path.exists(out + ".json"):
        return out, "skip"
    s, masks, role, _ = load_npz(path)
    key = f"{ds}|{det}|{seed}|{cat}|{tag}"
    with Timer() as tm:
        res = evaluate_rules(s, masks, role, mode=mode, frac=frac, R=R,
                             seed=int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2**31)
    rec = run_record({"tag": tag, "scores": path, "mode": mode, "frac": frac, "R": R})
    rec.update(res)
    rec["scores_sha256"] = sha256_file(path)
    rec["seconds"] = tm.seconds
    os.makedirs(os.path.dirname(out), exist_ok=True)
    atomic_json(rec, out + ".json")
    return out, f"{tm.seconds:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frontier", "contam"], required=True)
    ap.add_argument("--frac", type=float, nargs="+", default=[0.0])
    ap.add_argument("--glob", default="scores/*/*/seed0/*.npz")
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.glob))
    fracs = [0.0] if a.mode == "frontier" else a.frac
    jobs = [(p, a.mode, f, a.R) for f in fracs for p in paths]
    with Pool(a.workers) as pool:
        for out, msg in pool.imap_unordered(job, jobs):
            print(out, msg, flush=True)


if __name__ == "__main__":
    main()
