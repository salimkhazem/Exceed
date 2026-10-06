"""Robustness records written with the evaluator before its speed fix (commit cabf94c, kept in
results/robust_oldcode/) must be reproduced exactly by the final evaluator (results/raw/).
Writes results/summaries/robust_reproduction.json; exits 1 on any mismatch or missing record.
BTAD records of the old run have no counterpart (BTAD is not part of these experiments)."""
import glob
import json
import os
import sys


def main():
    same, diff, missing = [], [], []
    for old in sorted(glob.glob("results/robust_oldcode/*/*/*/*/*.json")):
        if os.sep + "btad" + os.sep in old:
            continue
        new = old.replace("results/robust_oldcode/", "results/raw/")
        if not os.path.exists(new):
            missing.append(new)
            continue
        a, b = json.load(open(old))["summary"], json.load(open(new))["summary"]
        (same if a == b else diff).append(new)
    out = {"checked": len(same) + len(diff) + len(missing), "identical": len(same),
           "different": diff, "missing": missing}
    os.makedirs("results/summaries", exist_ok=True)
    with open("results/summaries/robust_reproduction.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"robustness records re-run with the final evaluator: {out['checked']}; identical: {len(same)}; "
          f"different: {len(diff)}; missing: {len(missing)}")
    return 0 if not diff and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
