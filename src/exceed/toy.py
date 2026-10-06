"""Synthetic score maps with known ground truth (paper Sec. 6.1 and Prop. 3).

`field` draws a nominal score map: a location-dependent mean/scale (non-exchangeable
locations) plus a spatially smoothed Gaussian field, optionally shifted by an image-level
random effect. `anomalous` adds a disc-shaped defect. `hot_pixel_law` is the
intensity-extent coupled nominal law of Prop. 3 (hotter artifacts cover fewer units).
"""
from __future__ import annotations

import numpy as np


def _kernel(sigma: float, half: int = 6) -> np.ndarray:
    x = np.arange(-half, half + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _blur(z: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, z)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, z)


class ToyFields:
    def __init__(self, size: int = 64, sigma: float = 2.0, hetero: bool = True,
                 global_sd: float = 0.0, seed: int = 0):
        self.H = self.W = size
        self.rng = np.random.default_rng(seed)
        self.k = _kernel(sigma)
        self.mu = np.zeros((size, size))
        self.sd = np.ones((size, size))
        if hetero:
            self.mu[size // 12: size // 4, size // 12: size // 4] = 2.5
            self.sd[int(0.62 * size): int(0.94 * size), : size // 3] = 2.0
        self.global_sd = global_sd

    def field(self) -> np.ndarray:
        z = _blur(self.rng.standard_normal((self.H, self.W)), self.k)
        g = self.rng.normal(0.0, self.global_sd) if self.global_sd > 0 else 0.0
        return self.mu + self.sd * z / z.std() + g

    def anomalous(self, amp: float, rmin: int = 2, rmax: int = 7):
        s = self.field()
        r = int(self.rng.integers(rmin, rmax))
        cy, cx = self.rng.integers(r + self.H // 4, self.H - r, 2)
        yy, xx = np.ogrid[: self.H, : self.W]
        A = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        return s + amp * A, A


def hot_pixel_law(m: int, alpha: float, rng: np.random.Generator):
    """One nominal image of the Prop. 3 law, returned compactly as (level L, #hot K).
    L ~ U(0, L_max) with K(L) = round(m r^L), ln(1/r) = 2/alpha and
    L_max = 0.95 ln(m)/ln(1/r) so that K >= 1 without flooring. Hot units score 1 + L;
    the other units score in [0, 1)."""
    lr = 2.0 / alpha
    lmax = 0.95 * np.log(m) / lr
    L = rng.uniform(0.0, lmax)
    K = max(1, int(round(m * np.exp(-lr * L))))
    return L, K
