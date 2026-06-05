"""
submission.py — Space Data Task 3: Lunar Image Denoising
=========================================================
Models and weights: https://huggingface.co/mattiademartino/unet-deep-lunar-denoiser

Best model: unet_deep_01
Architecture: 4-stage UNetDeep with features [64, 128, 256, 512]
Training: MSE loss, AdamW, cosine-annealing LR schedule, 150 epochs, 15% val split
Results: PSNR 27.55 dB  |  MSE 0.001758  |  MAE 0.03213

Alternative: Conditional DDPM diffusion model (diffusion_01)
Architecture: Conditional U-Net backbone with sinusoidal timestep embedding, DDIM inference
Training: x0-prediction MSE loss, AdamW, cosine LR, 150 epochs, 15% val split
Results: PSNR 25.76 dB  |  MSE 0.002654  |  MAE 0.03989

Usage
-----
  # Train unet_deep_01 from scratch
  python submission.py train --data-dir data/ --out-dir results/submission/

  # Train diffusion_01 from scratch
  python submission.py train --model diffusion --data-dir data/ --out-dir results/submission_diffusion/

  # Run inference with UNet downloading from Hugging Face
  python3 submission.py test --data-dir data/ --hf-model mattiademartino/unet-deep-lunar-denoiser --out predictions.npy

  # Run inference with diffusion model download from Hugging Face
  python3 submission.py test --model diffusion --hf-model mattiademartino/unet-deep-lunar-denoiser --out predictions_diffusion.npy
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Soft spatial attention gate (Oktay et al. 2018)."""
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_x  = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi  = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


# ---------------------------------------------------------------------------
# Model 1: UNetDeep  (best model — unet_deep_01)
# ---------------------------------------------------------------------------

class UNetDeep(nn.Module):
    """4-stage U-Net; bottleneck at 4x4 for 64x64 inputs."""

    def __init__(
        self,
        features: Sequence[int] = (64, 128, 256, 512),
        dropout: float = 0.10887,
    ):
        super().__init__()
        f = list(features)
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(1, f[0], dropout)
        self.enc2 = DoubleConv(f[0], f[1], dropout)
        self.enc3 = DoubleConv(f[1], f[2], dropout)
        self.enc4 = DoubleConv(f[2], f[3], dropout)
        self.bottleneck = DoubleConv(f[3], f[3] * 2, dropout)
        self.up4 = nn.ConvTranspose2d(f[3] * 2, f[3], 2, stride=2)
        self.dec4 = DoubleConv(f[3] * 2, f[3], dropout)
        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2] * 2, f[2], dropout)
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1] * 2, f[1], dropout)
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0] * 2, f[0], dropout)
        self.out_conv = nn.Conv2d(f[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


def build_unet() -> UNetDeep:
    """Return unet_deep_01 with its exact HPO hyperparameters."""
    return UNetDeep(features=[64, 128, 256, 512], dropout=0.10887)


# ---------------------------------------------------------------------------
# Model 2: Conditional DDPM  (diffusion_01)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeEmb(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            SinusoidalPosEmb(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class TimeResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_ch)
        self.time_proj = nn.Linear(t_dim, out_ch * 2)
        self.residual = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        scale, shift = self.time_proj(t_emb).chunk(2, dim=1)
        h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class DiffusionAttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(f_g, f_int, 1), _group_norm(f_int))
        self.W_x  = nn.Sequential(nn.Conv2d(f_l, f_int, 1), _group_norm(f_int))
        self.psi  = nn.Sequential(nn.Conv2d(f_int, 1, 1), _group_norm(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class CondDDPMUNet(nn.Module):
    """
    Conditional U-Net backbone for the diffusion denoiser (diffusion_01).

    Input: 2-channel tensor [x_t || y] where x_t is the diffusion state and
    y is the sensor-noisy conditioning image. Timestep t is injected into the
    residual blocks via a sinusoidal + MLP embedding.
    Output: predicted clean image x0 in [0, 1].
    """

    def __init__(self, features: Sequence[int] = (32, 64, 128, 256), t_dim: int = 128):
        super().__init__()
        f = list(features)
        self.pool = nn.MaxPool2d(2)

        self.time_emb = TimeEmb(t_dim, t_dim, hidden_dim=512)

        self.enc1       = TimeResBlock(2, f[0], t_dim)
        self.enc2       = TimeResBlock(f[0], f[1], t_dim)
        self.enc3       = TimeResBlock(f[1], f[2], t_dim)
        self.enc4       = TimeResBlock(f[2], f[3], t_dim)
        self.bottleneck = TimeResBlock(f[3], f[3] * 2, t_dim)

        self.up4  = nn.ConvTranspose2d(f[3] * 2, f[3], 2, stride=2)
        self.att4 = DiffusionAttentionGate(f[3], f[3], max(f[3] // 2, 1))
        self.dec4 = TimeResBlock(f[3] * 2, f[3], t_dim)

        self.up3  = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.att3 = DiffusionAttentionGate(f[2], f[2], max(f[2] // 2, 1))
        self.dec3 = TimeResBlock(f[2] * 2, f[2], t_dim)

        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.att2 = DiffusionAttentionGate(f[1], f[1], max(f[1] // 2, 1))
        self.dec2 = TimeResBlock(f[1] * 2, f[1], t_dim)

        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.att1 = DiffusionAttentionGate(f[0], f[0], max(f[0] // 2, 1))
        self.dec1 = TimeResBlock(f[0] * 2, f[0], t_dim)

        self.out_conv = nn.Conv2d(f[0], 1, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        y:   torch.Tensor,
        t:   torch.Tensor,
    ) -> torch.Tensor:
        x  = torch.cat([x_t, y], dim=1)
        t_emb = self.time_emb(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        e4 = self.enc4(self.pool(e3), t_emb)
        b  = self.bottleneck(self.pool(e4), t_emb)
        g4 = self.up4(b)
        d4 = self.dec4(torch.cat([g4, self.att4(g4, e4)], dim=1), t_emb)
        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], dim=1), t_emb)
        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1), t_emb)
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1), t_emb)
        return torch.sigmoid(self.out_conv(d1))


class GaussianDiffusion(nn.Module):
    """
    Gaussian DDPM wrapper around CondDDPMUNet.

    Forward process:  x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε
    Training:         x0-prediction with random t
    Inference:        DDIM deterministic sampling (η=0), n_steps << T
    """

    def __init__(
        self,
        model:      CondDDPMUNet,
        T:          int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.T     = T

        betas     = torch.linspace(beta_start, beta_end, T)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas",             betas)
        self.register_buffer("alpha_bar",         alpha_bar)
        self.register_buffer("sqrt_ab",           alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_ab", (1.0 - alpha_bar).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sq_ab   = self.sqrt_ab[t].view(-1, 1, 1, 1)
        sq_omab = self.sqrt_one_minus_ab[t].view(-1, 1, 1, 1)
        return sq_ab * x0 + sq_omab * noise

    def training_step(self, x0: torch.Tensor, y: torch.Tensor, criterion) -> torch.Tensor:
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        x0_pred = self.model(self.q_sample(x0, t), y, t)
        return criterion(x0_pred, x0)

    @torch.no_grad()
    def ddim_sample(self, y: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """DDIM deterministic reverse sampling (η=0)."""
        B, _, H, W = y.shape
        device = y.device
        ts  = torch.linspace(self.T - 1, 0, n_steps + 1, dtype=torch.long, device=device)
        x_t = torch.randn(B, 1, H, W, device=device)

        for i in range(n_steps):
            t_val  = int(ts[i].item())
            t_next = int(ts[i + 1].item())
            t_b    = torch.full((B,), t_val, dtype=torch.long, device=device)

            x0_hat   = self.model(x_t, y, t_b).clamp(0.0, 1.0)
            ab_cur   = self.alpha_bar[t_val]
            ab_next  = self.alpha_bar[t_next]
            pred_eps = (x_t - ab_cur.sqrt() * x0_hat) / (1.0 - ab_cur).sqrt()
            x_t      = ab_next.sqrt() * x0_hat + (1.0 - ab_next).sqrt() * pred_eps

        return x_t.clamp(0.0, 1.0)


def build_diffusion() -> GaussianDiffusion:
    """Return diffusion_01 with its exact HPO hyperparameters."""
    unet = CondDDPMUNet(features=[32, 64, 128, 256], t_dim=128)
    return GaussianDiffusion(unet, T=1000)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _to_float32(arr: np.ndarray) -> np.ndarray:
    data = arr.astype(np.float32)
    if data.max() > 2.0:
        data /= 255.0
    return data


class LunarDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray):
        self.noisy = torch.from_numpy(_to_float32(noisy)).unsqueeze(1)
        self.clean = torch.from_numpy(_to_float32(clean)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, idx: int):
        return self.noisy[idx], self.clean[idx]


class LunarTestDataset(Dataset):
    def __init__(self, noisy: np.ndarray):
        self.noisy = torch.from_numpy(_to_float32(noisy)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.noisy[idx]


def _load_split(data_dir: Path, val_split: float = 0.15, seed: int = 42):
    noisy   = np.load(data_dir / "noisy_train_19k_harder.npy")
    clean   = np.load(data_dir / "clean_train_19k_harder.npy")
    dataset = LunarDataset(noisy, clean)
    val_sz  = int(val_split * len(dataset))
    gen     = torch.Generator().manual_seed(seed)
    return random_split(dataset, [len(dataset) - val_sz, val_sz], generator=gen)


# ---------------------------------------------------------------------------
# Shared metric helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _val_metrics(model_fn, loader: DataLoader, device: torch.device):
    """Returns (mse, mae, psnr_dB). model_fn(noisy_batch) → prediction."""
    mse_fn  = nn.MSELoss()
    mae_fn  = nn.L1Loss()
    mse_sum = mae_sum = n = 0.0
    for noisy_b, clean_b in loader:
        noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
        pred = model_fn(noisy_b)
        bs   = noisy_b.size(0)
        mse_sum += mse_fn(pred, clean_b).item() * bs
        mae_sum += mae_fn(pred, clean_b).item() * bs
        n       += bs
    mse  = mse_sum / n
    mae  = mae_sum / n
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    return mse, mae, psnr


# ---------------------------------------------------------------------------
# Train — UNet (unet_deep_01)
# ---------------------------------------------------------------------------

UNET_EPOCHS      = 150
UNET_BATCH_SIZE  = 16
UNET_LR          = 9.664e-4
UNET_WEIGHT_DECAY = 2.472e-3


def train(data_dir: str | Path = "data", out_dir: str | Path = "results/submission") -> Path:
    """
    Train unet_deep_01 from scratch.

    Parameters
    ----------
    data_dir : directory with noisy_train_19k_harder.npy and clean_train_19k_harder.npy
    out_dir  : where to save best_model_mae.pt, best_model_mse.pt, final_model.pt

    Returns
    -------
    Path to best_model_mae.pt
    """
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[unet_deep_01] Device: {device}")

    train_ds, val_ds = _load_split(data_dir)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=UNET_BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=UNET_BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = build_unet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=UNET_LR, weight_decay=UNET_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UNET_EPOCHS)

    best_mse = best_mae = float("inf")

    for epoch in range(1, UNET_EPOCHS + 1):
        model.train()
        tr_loss = tr_n = 0.0
        for noisy_b, clean_b in train_loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            loss = criterion(model(noisy_b), clean_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * noisy_b.size(0)
            tr_n    += noisy_b.size(0)
        scheduler.step()

        val_mse, val_mae, val_psnr = _val_metrics(model, val_loader, device)

        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), out_dir / "best_model_mse.pt")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), out_dir / "best_model_mae.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{epoch:3d}/{UNET_EPOCHS}]  train {tr_loss/tr_n:.6f}  "
                  f"val_mse {val_mse:.6f}  val_mae {val_mae:.6f}  PSNR {val_psnr:.2f} dB")

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    print(f"\nBest val MSE: {best_mse:.6f}  Best val MAE: {best_mae:.6f}")
    print(f"Weights saved to {out_dir}/")
    return out_dir / "best_model_mae.pt"


# ---------------------------------------------------------------------------
# Train — Diffusion (diffusion_01)
# ---------------------------------------------------------------------------

DIFF_EPOCHS      = 150
DIFF_BATCH_SIZE  = 32
DIFF_LR          = 4.248e-4
DIFF_WEIGHT_DECAY = 4.583e-5
DIFF_DDIM_STEPS  = 20   # steps used during training-time validation
VAL_EVERY        = 10   # validate every N epochs (DDIM is slow)


def train_diffusion(data_dir: str | Path = "data",
                    out_dir: str | Path = "results/submission_diffusion") -> Path:
    """
    Train diffusion_01 (Conditional DDPM) from scratch.

    Parameters
    ----------
    data_dir : directory with noisy_train_19k_harder.npy and clean_train_19k_harder.npy
    out_dir  : where to save best_model_mae.pt, best_model_mse.pt, final_model.pt

    Returns
    -------
    Path to best_model_mae.pt
    """
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diffusion_01] Device: {device}")

    train_ds, val_ds = _load_split(data_dir)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=DIFF_BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=DIFF_BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    diffusion = build_diffusion().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(diffusion.parameters(), lr=DIFF_LR, weight_decay=DIFF_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DIFF_EPOCHS)

    best_mse = best_mae = float("inf")

    for epoch in range(1, DIFF_EPOCHS + 1):
        diffusion.model.train()
        ep_loss = n_b = 0.0
        for noisy_b, clean_b in train_loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            loss = diffusion.training_step(clean_b, noisy_b, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_b     += 1
        scheduler.step()

        if epoch % VAL_EVERY == 0 or epoch == DIFF_EPOCHS:
            diffusion.model.eval()
            def _infer(y): return diffusion.ddim_sample(y, n_steps=DIFF_DDIM_STEPS)
            val_mse, val_mae, val_psnr = _val_metrics(_infer, val_loader, device)

            if val_mse < best_mse:
                best_mse = val_mse
                torch.save(diffusion.state_dict(), out_dir / "best_model_mse.pt")
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(diffusion.state_dict(), out_dir / "best_model_mae.pt")

            print(f"  [{epoch:3d}/{DIFF_EPOCHS}]  train {ep_loss/n_b:.5f}  "
                  f"val_mse {val_mse:.6f}  val_mae {val_mae:.6f}  PSNR {val_psnr:.2f} dB")

    torch.save(diffusion.state_dict(), out_dir / "final_model.pt")
    print(f"\nBest val MSE: {best_mse:.6f}  Best val MAE: {best_mae:.6f}")
    print(f"Weights saved to {out_dir}/")
    return out_dir / "best_model_mae.pt"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _resolve_weights_path(weights_path: str | Path) -> Path:
    """Return path to weights file, searching the project tree if not found at the given path."""
    p = Path(weights_path)
    if p.exists():
        return p
    root = Path(__file__).parent
    matches = sorted(root.rglob(p.name))
    if matches:
        found = matches[0]
        print(f"  (found weights at {found})")
        return found
    raise FileNotFoundError(
        f"Could not find '{p.name}'. "
        "Pass the full path with --weights or use --hf-model to download from Hugging Face."
    )


def _load_unet_weights(weights_path: str | Path | None,
                       hf_model: str | None,
                       device: torch.device) -> UNetDeep:
    model = build_unet().to(device)
    state = _fetch_state(weights_path, hf_model, ("best_model_mae.pt", "best_model.pt"), device)
    model.load_state_dict(state)
    return model


def _load_diffusion_weights(weights_path: str | Path | None,
                            hf_model: str | None,
                            device: torch.device) -> GaussianDiffusion:
    diff  = build_diffusion().to(device)
    state = _fetch_state(weights_path, hf_model, ("best_model.pt", "best_model_diff.pt"), device)
    diff.load_state_dict(state)
    return diff


def _fetch_state(weights_path, hf_model, hf_filename, device):
    if weights_path is not None:
        resolved = _resolve_weights_path(weights_path)
        print(f"Loading local weights: {resolved}")
        state = torch.load(resolved, map_location=device)
    elif hf_model is not None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            sys.exit("Install huggingface_hub:  pip install huggingface_hub")
        hf_filenames = (hf_filename,) if isinstance(hf_filename, str) else tuple(hf_filename)
        for idx, filename in enumerate(hf_filenames):
            try:
                print(f"Downloading weights from Hugging Face: {hf_model}/{filename}")
                local_pt = hf_hub_download(repo_id=hf_model, filename=filename)
                break
            except Exception:
                if idx == len(hf_filenames) - 1:
                    raise
                print(f"  (not found: {filename}; trying {hf_filenames[idx + 1]})")
        state    = torch.load(local_pt, map_location=device)
    else:
        sys.exit("Provide either --weights or --hf-model")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return state


def _save_preview_images(noisy: np.ndarray, predictions: np.ndarray,
                         preview_dir: Path, num_preview: int) -> Path | None:
    if num_preview <= 0:
        return None
    import matplotlib.pyplot as plt
    preview_dir.mkdir(parents=True, exist_ok=True)
    n = min(num_preview, len(predictions))
    for idx in range(n):
        pred_u8 = (np.clip(predictions[idx], 0.0, 1.0) * 255).astype(np.uint8)
        plt.imsave(preview_dir / f"prediction_{idx:04d}.png", pred_u8, cmap="gray",
                   vmin=0, vmax=255)
    grid_n = min(n, 8)
    noisy_f = _to_float32(noisy)
    fig, axes = plt.subplots(2, grid_n, figsize=(2.0 * grid_n, 4.0))
    if grid_n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for idx in range(grid_n):
        axes[0, idx].imshow(noisy_f[idx], cmap="gray", vmin=0, vmax=1)
        axes[0, idx].set_title(f"Noisy {idx}")
        axes[1, idx].imshow(predictions[idx], cmap="gray", vmin=0, vmax=1)
        axes[1, idx].set_title(f"Pred {idx}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(preview_dir / "preview_grid.png"), dpi=160)
    plt.close(fig)
    return preview_dir.resolve()


# ---------------------------------------------------------------------------
# Test — UNet
# ---------------------------------------------------------------------------

def test(
    data_dir: str | Path = "data",
    weights_path: str | Path | None = "best_model.pt",
    hf_model: str | None = None,
    out_path: str | Path = "predictions.npy",
    batch_size: int = 64,
    preview_dir: str | Path | None = None,
    num_preview: int = 16,
) -> np.ndarray:
    """
    Run UNet inference on the blind test set.

    Parameters
    ----------
    data_dir     : directory containing noisy_val_500_harder.npy
    weights_path : path to a local .pt checkpoint (takes priority over hf_model)
    hf_model     : Hugging Face repo id, e.g. 'mattiademartino/unet-deep-lunar-denoiser'
    out_path     : where to write predictions (N, H, W) float32 in [0, 1]
    batch_size   : inference batch size
    preview_dir  : where to write PNG previews (default: <out>_preview_images)
    num_preview  : number of PNG previews to save; 0 to disable

    Returns
    -------
    numpy array of predictions, shape (N, H, W)
    """
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is None:
        preview_dir = out_path.parent / f"{out_path.stem}_preview_images"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[unet_deep_01] Device: {device}")

    model = _load_unet_weights(weights_path, hf_model, device)
    model.eval()

    noisy_test = np.load(data_dir / "noisy_val_500_harder.npy")
    loader     = DataLoader(LunarTestDataset(noisy_test),
                            batch_size=batch_size, shuffle=False, num_workers=4)

    preds = []
    with torch.no_grad():
        for noisy_b in loader:
            preds.append(model(noisy_b.to(device)).cpu().squeeze(1).numpy())

    predictions = np.concatenate(preds, axis=0)
    np.save(out_path, predictions)
    print(f"Saved predictions → {out_path.resolve()}  shape={predictions.shape}")
    _save_preview_images(noisy_test, predictions, Path(preview_dir), num_preview)
    return predictions


# ---------------------------------------------------------------------------
# Test — Diffusion
# ---------------------------------------------------------------------------

def test_diffusion(
    data_dir: str | Path = "data",
    weights_path: str | Path | None = None,
    hf_model: str | None = None,
    out_path: str | Path = "predictions_diffusion.npy",
    ddim_steps: int = 50,
    batch_size: int = 32,
    preview_dir: str | Path | None = None,
    num_preview: int = 16,
) -> np.ndarray:
    """
    Run Conditional DDPM inference (DDIM) on the blind test set.

    Parameters
    ----------
    data_dir     : directory containing noisy_val_500_harder.npy
    weights_path : path to a local .pt checkpoint of the full GaussianDiffusion module
    hf_model     : Hugging Face repo id (downloads diffusion best_model.pt automatically)
    out_path     : where to write predictions (N, H, W) float32 in [0, 1]
    ddim_steps   : number of DDIM reverse steps (default 50; 20 is faster but slightly worse)
    batch_size   : inference batch size
    preview_dir  : where to write PNG previews
    num_preview  : number of PNG previews to save; 0 to disable

    Returns
    -------
    numpy array of predictions, shape (N, H, W)
    """
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is None:
        preview_dir = out_path.parent / f"{out_path.stem}_preview_images"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diffusion_01] Device: {device}  DDIM steps: {ddim_steps}")

    diffusion = _load_diffusion_weights(weights_path, hf_model, device)
    diffusion.model.eval()

    noisy_test = np.load(data_dir / "noisy_val_500_harder.npy")
    loader     = DataLoader(LunarTestDataset(noisy_test),
                            batch_size=batch_size, shuffle=False, num_workers=4)

    preds = []
    for noisy_b in loader:
        pred = diffusion.ddim_sample(noisy_b.to(device), n_steps=ddim_steps)
        preds.append(pred.cpu().squeeze(1).numpy())

    predictions = np.concatenate(preds, axis=0)
    np.save(out_path, predictions)
    print(f"Saved predictions → {out_path.resolve()}  shape={predictions.shape}")
    _save_preview_images(noisy_test, predictions, Path(preview_dir), num_preview)
    return predictions


# ---------------------------------------------------------------------------
# Hugging Face upload
# ---------------------------------------------------------------------------

def upload_to_hf(weights_path: str | Path, hf_repo: str) -> None:
    """Upload a .pt checkpoint to a Hugging Face model repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("Install huggingface_hub:  pip install huggingface_hub")

    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
    weights_path = Path(weights_path)
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo=weights_path.name,
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"Upload {weights_path.name}",
    )
    print(f"Uploaded {weights_path} → https://huggingface.co/{hf_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lunar denoising — submission script (unet_deep_01 + diffusion_01)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train a model from scratch")
    p_train.add_argument("--model", choices=["unet", "diffusion"], default="unet",
                         help="Which model to train (default: unet)")
    p_train.add_argument("--data-dir", default="data")
    p_train.add_argument("--out-dir",  default="results/submission")

    # --- test ---
    p_test = sub.add_parser("test", help="Run inference on the test set")
    p_test.add_argument("--model", choices=["unet", "diffusion"], default="unet",
                        help="Which model to use for inference (default: unet)")
    p_test.add_argument("--data-dir",    default="data")
    p_test.add_argument("--weights",     default=None,
                        help="Path to a local .pt checkpoint")
    p_test.add_argument("--hf-model",    default=None,
                        help="Hugging Face repo id, e.g. mattiademartino/unet-deep-lunar-denoiser")
    p_test.add_argument("--out",         default="predictions.npy")
    p_test.add_argument("--batch-size",  type=int, default=64)
    p_test.add_argument("--ddim-steps",  type=int, default=50,
                        help="DDIM reverse steps for diffusion inference (default: 50)")
    p_test.add_argument("--preview-dir", default=None)
    p_test.add_argument("--num-preview", type=int, default=16)

    # --- upload ---
    p_up = sub.add_parser("upload", help="Upload weights to Hugging Face Hub")
    p_up.add_argument("--weights",  required=True)
    p_up.add_argument("--hf-repo",  required=True,
                      help="e.g. mattiademartino/unet-deep-lunar-denoiser")

    args = parser.parse_args()

    if args.cmd == "train":
        if args.model == "unet":
            train(data_dir=args.data_dir, out_dir=args.out_dir)
        else:
            train_diffusion(data_dir=args.data_dir, out_dir=args.out_dir)

    elif args.cmd == "test":
        if args.model == "unet":
            test(
                data_dir=args.data_dir,
                weights_path=args.weights,
                hf_model=args.hf_model,
                out_path=args.out,
                batch_size=args.batch_size,
                preview_dir=args.preview_dir,
                num_preview=args.num_preview,
            )
        else:
            test_diffusion(
                data_dir=args.data_dir,
                weights_path=args.weights,
                hf_model=args.hf_model,
                out_path=args.out,
                ddim_steps=args.ddim_steps,
                batch_size=args.batch_size,
                preview_dir=args.preview_dir,
                num_preview=args.num_preview,
            )

    elif args.cmd == "upload":
        upload_to_hf(weights_path=args.weights, hf_repo=args.hf_repo)


if __name__ == "__main__":
    main()
