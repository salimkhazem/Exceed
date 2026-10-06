"""Training-free frozen-feature anomaly scorers producing (S x S) score maps.

- PatchCore-style: WideResNet-50 layer2+layer3 locally aggregated patch features, exact
  1-NN distance to a memory bank of fit-split nominal patches (no coreset: exact search).
- DINOv2 patch kNN: ViT-B/14 last-layer patch tokens, cosine distance to the bank.
- PaDiM-style: ResNet-18 layer1-3 features (random 100-channel subset), per-location
  Gaussian, Mahalanobis distance (location-specific nominal laws by construction).
All maps are bilinearly upsampled to S x S and Gaussian-smoothed (sigma = 4 px).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image(path: str, size: int) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return x


def load_mask(path: str | None, size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=bool)
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(m) > 0


def gaussian_blur(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    half = int(4 * sigma + 0.5)
    x = torch.arange(-half, half + 1, device=maps.device, dtype=maps.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    m = maps.unsqueeze(1)
    m = F.conv2d(F.pad(m, (half, half, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, half, half), mode="reflect"), k.view(1, 1, -1, 1))
    return m.squeeze(1)


class _Base:
    size = 256
    input_size = 256

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)

    def _batch(self, paths):
        x = torch.stack([load_image(p, self.input_size) for p in paths])
        return ((x - MEAN) / STD).to(self.device)

    @torch.no_grad()
    def score(self, paths, bs=8):
        return np.concatenate([self.score_tensor(self._batch(paths[i:i + bs]))
                               for i in range(0, len(paths), bs)])

    def _finish(self, grid_maps: torch.Tensor) -> np.ndarray:
        m = F.interpolate(grid_maps.unsqueeze(1).float(), size=(self.size, self.size),
                          mode="bilinear", align_corners=False).squeeze(1)
        return gaussian_blur(m).cpu().numpy().astype(np.float32)


class PatchCore(_Base):
    name = "patchcore"

    def __init__(self, device="cuda", bank_max=300_000, seed=0):
        super().__init__(device)
        import timm
        # torchvision IMAGENET1K_V1 weights, hosted on the HF hub by timm
        self.net = timm.create_model("wide_resnet50_2.tv_in1k", pretrained=True, features_only=True,
                                     out_indices=(2, 3)).eval().to(self.device)
        self.bank_max = bank_max
        self.seed = seed
        self.bank = None

    @torch.no_grad()
    def embed(self, x):
        f2, f3 = self.net(x)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear",
                           align_corners=False)
        f = torch.cat([f2, f3], 1)                       # (B, 1536, 32, 32)
        return f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1]), f.shape[-2:]

    @torch.no_grad()
    def fit(self, paths, bs=16):
        feats = []
        for i in range(0, len(paths), bs):
            e, self.grid = self.embed(self._batch(paths[i:i + bs]))
            feats.append(e.reshape(-1, e.shape[-1]).half())
        bank = torch.cat(feats)
        if bank.shape[0] > self.bank_max:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            idx = torch.randperm(bank.shape[0], generator=g)[: self.bank_max].to(bank.device)
            bank = bank[idx]
        self.bank = bank

    @torch.no_grad()
    def score_tensor(self, x, chunk=60_000):
        e, grid = self.embed(x)
        B, P, D = e.shape
        q = e.reshape(-1, D).float()
        best = torch.full((q.shape[0],), float("inf"), device=self.device)
        for j in range(0, self.bank.shape[0], chunk):
            d = torch.cdist(q, self.bank[j:j + chunk].float())
            best = torch.minimum(best, d.min(1).values)
        return self._finish(best.view(B, *grid))


class DinoKNN(_Base):
    name = "dinov2"
    input_size = 448                                     # 32 x 32 patch grid

    def __init__(self, device="cuda", bank_max=300_000, seed=0):
        super().__init__(device)
        import timm
        self.net = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                                     img_size=self.input_size).eval().to(self.device)
        self.bank_max = bank_max
        self.seed = seed

    @torch.no_grad()
    def embed(self, x):
        t = self.net.forward_features(x)                 # (B, prefix + N, 768)
        n_pre = getattr(self.net, "num_prefix_tokens", 1)
        p = F.normalize(t[:, n_pre:], dim=-1)
        g = int(round(p.shape[1] ** 0.5))
        return p, (g, g)

    fit = PatchCore.fit

    @torch.no_grad()
    def score_tensor(self, x, chunk=60_000):
        e, grid = self.embed(x)
        B, P, D = e.shape
        q = e.reshape(-1, D).half()
        best = torch.full((q.shape[0],), -float("inf"), device=self.device)
        for j in range(0, self.bank.shape[0], chunk):
            sim = (q @ self.bank[j:j + chunk].T).float()
            best = torch.maximum(best, sim.max(1).values)
        return self._finish((1.0 - best).view(B, *grid))


class PaDiM(_Base):
    name = "padim"

    def __init__(self, device="cuda", n_channels=100, seed=0):
        super().__init__(device)
        import timm
        self.net = timm.create_model("resnet18.tv_in1k", pretrained=True, features_only=True,
                                     out_indices=(1, 2, 3)).eval().to(self.device)
        g = torch.Generator().manual_seed(1234)
        self.sel = torch.randperm(448, generator=g)[:n_channels].to(self.device)

    @torch.no_grad()
    def embed(self, x):
        f1, f2, f3 = self.net(x)
        s = f1.shape[-2:]
        f = torch.cat([f1, F.interpolate(f2, size=s, mode="nearest"),
                       F.interpolate(f3, size=s, mode="nearest")], 1)
        return f[:, self.sel]                            # (B, C, 64, 64)

    @torch.no_grad()
    def fit(self, paths, bs=16):
        fs = torch.cat([self.embed(self._batch(paths[i:i + bs])).float()
                        for i in range(0, len(paths), bs)])   # (N, C, h, w)
        N, C, h, w = fs.shape
        x = fs.permute(2, 3, 0, 1).reshape(h * w, N, C)
        self.mu = x.mean(1)                                    # (hw, C)
        xc = x - self.mu[:, None]
        cov = xc.transpose(1, 2) @ xc / max(N - 1, 1)
        cov += 0.01 * torch.eye(C, device=self.device)
        self.prec = torch.linalg.inv(cov)                      # (hw, C, C)
        self.grid = (h, w)

    @torch.no_grad()
    def score_tensor(self, x):
        f = self.embed(x).float()
        B, C, h, w = f.shape
        z = f.permute(2, 3, 0, 1).reshape(h * w, B, C) - self.mu[:, None]
        d = torch.einsum("lbc,lcd,lbd->lb", z, self.prec, z).clamp_min(0).sqrt()
        return self._finish(d.T.reshape(B, h, w))


DETECTORS = {"patchcore": PatchCore, "dinov2": DinoKNN, "padim": PaDiM}
