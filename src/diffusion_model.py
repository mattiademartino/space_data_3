"""
Conditional DDPM for image denoising.

Architecture : Attention U-Net, 2-channel input (x_t concatenated with y),
               sinusoidal timestep embedding injected additively at bottleneck.
Parameterisation: x0-prediction (model predicts clean image directly).
Inference         : DDIM deterministic sampling in n_steps << T steps.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps (Vaswani et al.)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) long tensor of timestep indices
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )                                            # (half,)
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)     # (B, dim)


class TimeEmb(nn.Module):
    """Two-layer MLP mapping sinusoidal embedding to feature-map channels."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)   # (B, out_dim)


# ---------------------------------------------------------------------------
# Building blocks (shared with models.py style)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

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
# Conditional DDPM U-Net
# ---------------------------------------------------------------------------

class CondDDPMUNet(nn.Module):
    """
    Conditional U-Net backbone for the diffusion denoiser.

    The model takes:
      - x_t  : the image at diffusion timestep t  (1 channel)
      - y    : the sensor-noisy conditioning image (1 channel)
      - t    : the integer timestep

    x_t and y are concatenated along the channel axis to form a 2-channel
    input.  The timestep t is embedded via a sinusoidal + MLP embedding and
    added to the bottleneck feature map.

    The output is the predicted clean image x0 in [0, 1] (sigmoid activation).
    """

    def __init__(self, features: Sequence[int] = (32, 64, 128), t_dim: int = 128):
        super().__init__()
        f = list(features)
        self.pool = nn.MaxPool2d(2)

        # 2-channel input: [x_t  ||  y]
        self.enc1       = DoubleConv(2, f[0])
        self.enc2       = DoubleConv(f[0], f[1])
        self.enc3       = DoubleConv(f[1], f[2])
        self.bottleneck = DoubleConv(f[2], f[2] * 2)

        # Timestep embedding projected to bottleneck channels
        self.time_emb = TimeEmb(t_dim, f[2] * 2)

        # Decoder with attention gates on every skip connection
        self.up3  = nn.ConvTranspose2d(f[2] * 2, f[2], 2, stride=2)
        self.att3 = AttentionGate(f[2], f[2], max(f[2] // 2, 1))
        self.dec3 = DoubleConv(f[2] * 2, f[2])

        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.att2 = AttentionGate(f[1], f[1], max(f[1] // 2, 1))
        self.dec2 = DoubleConv(f[1] * 2, f[1])

        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.att1 = AttentionGate(f[0], f[0], max(f[0] // 2, 1))
        self.dec1 = DoubleConv(f[0] * 2, f[0])

        self.out_conv = nn.Conv2d(f[0], 1, 1)

    def forward(
        self,
        x_t: torch.Tensor,   # (B, 1, H, W)  diffusion state at step t
        y:   torch.Tensor,   # (B, 1, H, W)  sensor conditioning image
        t:   torch.Tensor,   # (B,) long     current diffusion timestep
    ) -> torch.Tensor:
        x = torch.cat([x_t, y], dim=1)   # (B, 2, H, W)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        # Inject timestep: broadcast over spatial dims
        b = b + self.time_emb(t).unsqueeze(-1).unsqueeze(-1)   # (B, C, H, W)

        g3 = self.up3(b)
        d3 = self.dec3(torch.cat([g3, self.att3(g3, e3)], dim=1))

        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))

        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))

        return torch.sigmoid(self.out_conv(d1))   # (B, 1, H, W) in [0, 1]


# ---------------------------------------------------------------------------
# Gaussian Diffusion process
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """
    Wraps CondDDPMUNet with the forward/reverse diffusion process.

    Forward process (fixed, no parameters):
        x_t = sqrt(ᾱ_t) * x0  +  sqrt(1 - ᾱ_t) * ε,   ε ~ N(0, I)

    Training: at each step sample random t, corrupt x0 → x_t, ask the
    model to predict x0 back.  Loss is applied in pixel space on x0_pred vs x0.

    Inference: DDIM deterministic reverse sampling (Song et al. 2021).
    By setting the stochastic noise coefficient η = 0, DDIM reduces the
    number of required steps from T=1000 to n_steps ≈ 50 with minimal
    quality loss.
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

        betas     = torch.linspace(beta_start, beta_end, T)   # (T,)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)               # (T,)

        self.register_buffer("betas",             betas)
        self.register_buffer("alpha_bar",         alpha_bar)
        self.register_buffer("sqrt_ab",           alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_ab", (1.0 - alpha_bar).sqrt())

    # ------------------------------------------------------------------
    # Forward process  (used only during training)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Corrupt clean image x0 at timestep t (forward diffusion)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sq_ab   = self.sqrt_ab[t].view(-1, 1, 1, 1)
        sq_omab = self.sqrt_one_minus_ab[t].view(-1, 1, 1, 1)
        return sq_ab * x0 + sq_omab * noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, x0: torch.Tensor, y: torch.Tensor, criterion
    ) -> torch.Tensor:
        """
        One training step:
          1. Sample random timestep t for each image in the batch.
          2. Corrupt x0 → x_t via the forward process.
          3. Ask the model to predict x0 from (x_t, y, t).
          4. Compute criterion(x0_pred, x0).
        """
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        x_t     = self.q_sample(x0, t)
        x0_pred = self.model(x_t, y, t)
        return criterion(x0_pred, x0)

    # ------------------------------------------------------------------
    # Inference  (DDIM)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(self, y: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        Deterministic DDIM reverse sampling conditioned on sensor image y.

        Algorithm (η = 0):
          1. Start from x_T ~ N(0, I).
          2. For each step t → t_prev (evenly spaced from T-1 to 0):
               a. Predict x0: x0_hat = model(x_t, y, t)
               b. Predict ε:  ε_hat  = (x_t - √ᾱ_t · x0_hat) / √(1-ᾱ_t)
               c. Jump to t_prev:
                  x_{t_prev} = √ᾱ_{t_prev} · x0_hat + √(1-ᾱ_{t_prev}) · ε_hat

        y       : (B, 1, H, W)  sensor noisy image (conditioning)
        n_steps : number of reverse steps (default 50)
        Returns : denoised image (B, 1, H, W) in [0, 1]
        """
        B, _, H, W = y.shape
        device     = y.device

        # Evenly-spaced timestep sequence from T-1 down to 0
        ts = torch.linspace(self.T - 1, 0, n_steps + 1, dtype=torch.long, device=device)

        x_t = torch.randn(B, 1, H, W, device=device)

        for i in range(n_steps):
            t_val      = int(ts[i].item())
            t_next_val = int(ts[i + 1].item())

            t_batch = torch.full((B,), t_val, dtype=torch.long, device=device)

            x0_hat  = self.model(x_t, y, t_batch).clamp(0.0, 1.0)

            ab_cur  = self.alpha_bar[t_val]
            ab_next = self.alpha_bar[t_next_val]

            pred_eps = (x_t - ab_cur.sqrt() * x0_hat) / (1.0 - ab_cur).sqrt()
            x_t = ab_next.sqrt() * x0_hat + (1.0 - ab_next).sqrt() * pred_eps

        return x_t.clamp(0.0, 1.0)
