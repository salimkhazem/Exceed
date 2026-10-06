"""Download public data: MVTec AD (CC BY-NC-SA 4.0; HF mirror of the original layout)
and VisA (CC BY 4.0; official S3 archive). Writes a checksum manifest."""
import argparse
import json
import os
import subprocess
import sys
import tarfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from exceed.data import MVTEC, MVTEC_CATEGORIES, ROOT  # noqa: E402

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


def mvtec():
    from huggingface_hub import snapshot_download
    rev = snapshot_download(repo_id="foersben/mvtec-ad", repo_type="dataset", local_dir=MVTEC,
                            allow_patterns=[f"{c}/**" for c in MVTEC_CATEGORIES]
                            + ["license.txt", "readme.txt"], max_workers=32)
    return {"source": "hf:foersben/mvtec-ad", "local": os.path.relpath(rev, ROOT)}


def visa():
    d = os.path.join(ROOT, "visa")
    tar = os.path.join(d, "VisA_20220922.tar")
    if not os.path.exists(tar):
        os.makedirs(d, exist_ok=True)
        subprocess.check_call(["curl", "-sL", "-o", tar, VISA_URL])
    out = os.path.join(d, "VisA")
    if not os.path.exists(os.path.join(out, "split_csv")):
        with tarfile.open(tar) as tf:
            tf.extractall(out, filter="data")
    return {"source": VISA_URL, "local": os.path.relpath(out, ROOT), "tar_bytes": os.path.getsize(tar)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["mvtec", "visa"])
    a = ap.parse_args()
    path = "results/summaries/data_sources.json"
    info = json.load(open(path)) if os.path.exists(path) else {}
    if "mvtec" in a.which:
        info["mvtec"] = mvtec()
    if "visa" in a.which:
        info["visa"] = visa()
    os.makedirs("results/summaries", exist_ok=True)
    with open("results/summaries/data_sources.json", "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info, indent=1))
