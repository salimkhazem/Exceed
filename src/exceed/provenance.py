"""Run records and atomic result writes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_record(config: dict) -> dict:
    import numpy as np
    rec = {
        "utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": config,
        "argv": sys.argv,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain", "--untracked-files=no") not in ("", "unknown"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "host_hash": hashlib.sha1(socket.gethostname().encode()).hexdigest()[:8],
    }
    try:
        import torch
        rec["torch"] = torch.__version__
        rec["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return rec


class Timer:
    """Wall time plus peak GPU memory (if torch/CUDA is in use)."""

    def __enter__(self):
        self.t0 = time.time()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        self.peak_gpu_bytes = 0
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
        except ImportError:
            pass
        return False


def _check_finite(obj, path="root"):
    if isinstance(obj, float) and not math.isfinite(obj) and not math.isnan(obj):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_finite(v, f"{path}[{i}]")


def atomic_json(obj, path: str) -> None:
    _check_finite(obj)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
