"""Aggregate per-category summaries: mean over categories and number of categories where a
rule's realized FDR exceeds the level (point estimate) for each FDR notion."""
import argparse
import csv
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--dataset", default="*")
ap.add_argument("--out", default=None)
a = ap.parse_args()

METRICS = ["fdr_anom", "fdr_anom_b16", "fdr_anom_b24", "fdr_region_r0", "fdr_region_r8",
           "fa_nominal", "defect_detection", "image_detection", "pixel_tpr_pooled",
           "f1_anom_pooled"]
FDRS = ["fdr_anom", "fdr_anom_b16", "fdr_anom_b24", "fdr_region_r0", "fdr_region_r8",
        "fa_nominal"]
table = {}
for p in sorted(glob.glob(f"results/raw/{a.tag}/{a.dataset}/*/*/*.json")):
    if p.endswith(".influence.json"):
        continue
    rec = json.load(open(p))
    ds, det = p.split(os.sep)[-4], p.split(os.sep)[-3]
    for rule, s in rec["summary"].items():
        name, alpha = rule.split("@")
        base = name.split("|")[0].split("/")[0]
        table.setdefault((ds, det, base, alpha), []).append(s)

rows = []
for (ds, det, base, alpha), ss in sorted(table.items()):
    lvl = float(alpha) if alpha != "None" else 0.1
    row = {"dataset": ds, "detector": det, "rule": base, "alpha": alpha, "n_cat": len(ss)}
    for mname in METRICS:
        v = np.array([s.get(mname, np.nan) for s in ss], float)
        row[mname] = float(np.nanmean(v))
        if mname in FDRS:
            row[f"viol_{mname}"] = int(np.nansum(v > lvl))
    rows.append(row)

if a.out:
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
hdr = ["dataset", "detector", "rule", "alpha", "n_cat", "fdr_anom", "fdr_anom_b24", "fdr_region_r0",
       "fa_nominal", "defect_detection", "image_detection", "pixel_tpr_pooled"]
print(" ".join(f"{h[:12]:>12s}" for h in hdr) + "   viol(b24,reg0,FA)")
for r in rows:
    if r["alpha"] not in ("0.1", "None"):
        continue
    vals = [r[h] if isinstance(r[h], str) else (f"{r[h]:.3f}" if isinstance(r[h], float) else str(r[h]))
            for h in hdr]
    print(" ".join(f"{v[:12]:>12s}" for v in vals),
          f"  {r.get('viol_fdr_anom_b24', '-')}/{r.get('viol_fdr_region_r0', '-')}/{r.get('viol_fa_nominal', '-')}")
