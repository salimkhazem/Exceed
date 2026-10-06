"""BTAD (beanTech Anomaly Detection, CC BY-SA; Mishra et al., 2021) records in the format of
exceed.data.records. Kept in its own module so that the scoring code of the main benchmarks
is unchanged."""
import os

from .data import ROOT, _img_files

BTAD = os.path.join(ROOT, "btad", "BTech_Dataset_transformed")
BTAD_CATEGORIES = ["01", "02", "03"]


def btad_records(category: str, root: str = BTAD):
    base = os.path.join(root, category)
    recs = []
    for split in ("train", "test"):
        for p in _img_files(os.path.join(base, split, "ok")):
            recs.append((p, None, False, "good", split))
    for p in _img_files(os.path.join(base, "test", "ko")):
        stem = os.path.splitext(os.path.basename(p))[0]
        cands = [os.path.join(base, "ground_truth", "ko", stem + ext) for ext in (".png", ".bmp")]
        mp = next((c for c in cands if os.path.exists(c)), None)
        if mp is None:
            raise FileNotFoundError(cands[0])
        recs.append((p, mp, True, "ko", "test"))
    return recs
