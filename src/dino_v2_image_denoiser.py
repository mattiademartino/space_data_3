"""
DINOv2-based Lunar Image Denoiser
==================================
Self-contained file covering:
  - Model definition  (DINOv2ViT-L encoder + convolutional decoder)
  - Dataset / DataLoader helpers
  - Training loop with cosine LR schedule
  - Evaluation (MSE + PSNR)
  - CLI entry-point

Architecture overview
---------------------
Input:  (B, 1, 128, 128)  – grayscale noisy image

Encoder – facebook/dinov2-large (frozen by default, optionally fine-tuned)
  • DINOv2 expects 3-channel RGB input → the single channel is repeated 3×.
  • DINOv2-large uses a patch size of 14, so 128px → 9×9 = 81 patches (with
    register tokens ignored).  Patch tokens are reshaped into a spatial grid
    and used as the bottleneck feature map passed to the decoder.
  • Intermediate patch features from layers {6, 12, 18, 24} are extracted via
    forward hooks and used as multi-scale skip connections.

Decoder – four upsampling stages, each doubling spatial resolution:
  9×9 → 18×18 → 36×36 → 72×72 → 128×128  (final bilinear resize to exact size)
  Each stage: bilinear upsample → concat skip → DoubleConv block

Output: (B, 1, 128, 128) sigmoid-activated clean image prediction

Usage
-----
  # Train with default settings:
  python dinov2_denoiser.py --noisy data/noisy.npy --clean data/clean.npy

  # Freeze encoder, bigger batch, more epochs:
  python dinov2_denoiser.py \
      --noisy data/noisy.npy --clean data/clean.npy \
      --epochs 60 --lr 3e-4 --batch-size 16 --freeze-encoder

  # Unfreeze the last 6 transformer blocks:
  python dinov2_denoiser.py \
      --noisy data/noisy.npy --clean data/clean.npy \
      --unfreeze-blocks 6

  # Evaluate a saved checkpoint (no training):
  python dinov2_denoiser.py \
      --noisy data/noisy.npy --clean data/clean.npy \
      --eval-only --checkpoint results/best_model.pt
"""

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not found – loss plots and comparison images will be skipped.")


# ---------------------------------------------------------------------------
# Config defaults  (all overridable via CLI)
# ---------------------------------------------------------------------------

DINOV2_MODEL   = "facebook/dinov2-large"   # HuggingFace model ID
IMG_SIZE       = 128                        # spatial size expected by the pipeline
PATCH_SIZE     = 14                         # DINOv2-large native patch size
# After padding 128 to the nearest multiple of 14 (=  ceil(128/14)*14 = 140),
# we get  140/14 = 10 × 10 = 100 patch tokens.
# We pad during pre-processing and crop the feature map back to 128 after decoding.
PADDED_SIZE    = 140                        # ceil(128/14)*14
GRID_SIZE      = PADDED_SIZE // PATCH_SIZE  # 10 – spatial grid of patch tokens
EMBED_DIM      = 1024                       # DINOv2-large hidden dimension

# Intermediate layers from which we extract multi-scale features.
# DINOv2-large has 24 transformer blocks (0-indexed).
HOOK_LAYERS    = [5, 11, 17, 23]            # ~quarter-depth boundaries

# Decoder channel widths for the four upsampling stages (coarse → fine)
DECODER_CHANNELS = [512, 256, 128, 64]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _to_float32(arr: np.ndarray) -> np.ndarray:
    data = arr.astype(np.float32)
    if data.max() > 2.0:
        data /= 255.0
    return data


class LunarDataset(Dataset):
    """Paired noisy / clean dataset loaded from .npy files."""

    def __init__(self, noisy: np.ndarray, clean: np.ndarray):
        self.noisy = torch.from_numpy(_to_float32(noisy)).unsqueeze(1)  # (N,1,H,W)
        self.clean = torch.from_numpy(_to_float32(clean)).unsqueeze(1)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


def make_loaders(noisy_path: str, clean_path: str,
                 val_split: float = 0.15,
                 batch_size: int = 8,
                 seed: int = 42,
                 num_workers: int = 4):
    noisy = np.load(noisy_path)
    clean = np.load(clean_path)
    ds = LunarDataset(noisy, clean)

    val_size   = int(val_split * len(ds))
    train_size = len(ds) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds, val_ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Conv-BN-ReLU × 2, standard building block re-used from the project."""

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

    def forward(self, x):
        return self.block(x)


class ProjectionConv(nn.Module):
    """1×1 conv to project encoder features to a target channel count."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DINOv2Denoiser(nn.Module):
    """
    DINOv2-large encoder  +  four-stage convolutional decoder for denoising.

    Parameters
    ----------
    freeze_encoder : bool
        If True, all DINOv2 parameters are frozen (fast, good for small data).
    unfreeze_last_n : int
        Number of transformer blocks counted from the end to unfreeze even
        when freeze_encoder=True.  Useful for lightweight fine-tuning.
    """

    def __init__(self, freeze_encoder: bool = True, unfreeze_last_n: int = 0):
        super().__init__()

        # ---- encoder --------------------------------------------------------
        try:
            from transformers import Dinov2Model
        except ImportError as e:
            raise ImportError(
                "transformers library is required.  "
                "Install with: pip install transformers"
            ) from e

        self.encoder = Dinov2Model.from_pretrained(DINOV2_MODEL)
        self.encoder.config.output_hidden_states = True

        # Freeze / selectively unfreeze
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Optionally thaw the last N transformer blocks
            if unfreeze_last_n > 0:
                n_blocks = len(self.encoder.encoder.layer)
                for block in self.encoder.encoder.layer[n_blocks - unfreeze_last_n:]:
                    for p in block.parameters():
                        p.requires_grad = True
                # Also unfreeze the final layer norm so the unfrozen blocks
                # can adapt the representation that exits the encoder.
                for p in self.encoder.layernorm.parameters():
                    p.requires_grad = True

        # ---- skip-connection projections ------------------------------------
        # Hook layers [5, 11, 17, 23] all produce EMBED_DIM (1024) features.
        # We project each down to match the corresponding decoder stage's
        # skip-input width.
        self.skip_proj = nn.ModuleList([
            ProjectionConv(EMBED_DIM, ch)
            for ch in DECODER_CHANNELS          # 512, 256, 128, 64
        ])

        # ---- bottleneck projection ------------------------------------------
        # The last hidden state is also EMBED_DIM; project to the first
        # decoder stage channel count.
        self.bottleneck_proj = ProjectionConv(EMBED_DIM, DECODER_CHANNELS[0])

        # ---- decoder --------------------------------------------------------
        # Stage i receives: upsampled features (DECODER_CHANNELS[i]) +
        #                   projected skip    (DECODER_CHANNELS[i])
        # DoubleConv input width = 2 × DECODER_CHANNELS[i]
        self.decoder = nn.ModuleList()
        for i, ch in enumerate(DECODER_CHANNELS):
            in_ch = ch * 2   # upsampled + skip
            out_ch = DECODER_CHANNELS[i + 1] if i + 1 < len(DECODER_CHANNELS) else ch
            self.decoder.append(DoubleConv(in_ch, out_ch))

        # ---- output head ----------------------------------------------------
        self.out_conv = nn.Conv2d(DECODER_CHANNELS[-1], 1, kernel_size=1)

    # ------------------------------------------------------------------
    def _extract_features(self, pixel_values: torch.Tensor):
        """
        Run the DINOv2 encoder and return:
          bottleneck : (B, EMBED_DIM, GRID, GRID) spatial feature map
          skips       : list of 4 spatial maps at hook layers, coarse→fine
        """
        outputs = self.encoder(pixel_values=pixel_values,
                               output_hidden_states=True)

        # hidden_states is a tuple of length (n_layers + 1), each (B, 1+N_patches, D)
        # Index 0 is the embedding output before layer 0.
        hidden_states = outputs.hidden_states  # tuple, each (B, 1+N, D)

        def _to_spatial(h):
            # h: (B, 1 + GRID*GRID, D)  — drop the [CLS] token
            B, _, D = h.shape
            tokens = h[:, 1:, :]                         # (B, GRID*GRID, D)
            return tokens.permute(0, 2, 1).reshape(B, D, GRID_SIZE, GRID_SIZE)

        # HOOK_LAYERS are 0-indexed block indices; hidden_states[i+1] is the
        # output of block i (hidden_states[0] = pre-block embeddings).
        skips = [_to_spatial(hidden_states[layer_idx + 1])
                 for layer_idx in HOOK_LAYERS]            # coarse→fine order

        # Use the final hidden state as the bottleneck
        bottleneck = _to_spatial(hidden_states[-1])

        return bottleneck, skips

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, 128, 128)  grayscale noisy image in [0, 1]
        returns : (B, 1, 128, 128)  denoised image in [0, 1]
        """
        B, C, H, W = x.shape

        # --- pre-processing --------------------------------------------------
        # 1. Repeat grayscale channel → RGB
        x_rgb = x.repeat(1, 3, 1, 1)                    # (B, 3, H, W)

        # 2. Pad to nearest multiple of PATCH_SIZE (14): 128 → 140
        pad_h = PADDED_SIZE - H
        pad_w = PADDED_SIZE - W
        x_padded = F.pad(x_rgb, (0, pad_w, 0, pad_h), mode="reflect")  # (B,3,140,140)

        # 3. DINOv2 was pre-trained with ImageNet mean/std on 3-channel images.
        #    Apply the same normalisation so the frozen weights see expected inputs.
        mean = x_padded.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = x_padded.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_norm = (x_padded - mean) / std

        # --- encoder ---------------------------------------------------------
        bottleneck, skips = self._extract_features(x_norm)
        # bottleneck : (B, 1024, 10, 10)
        # skips[0..3]: (B, 1024, 10, 10)  — all same spatial size since
        #              DINOv2 is an isotropic ViT (no intermediate pooling)

        # --- bottleneck projection -------------------------------------------
        z = self.bottleneck_proj(bottleneck)             # (B, 512, 10, 10)

        # --- decoder ---------------------------------------------------------
        # Skips are ordered coarse→fine (layer 5 → 23); we consume them in that
        # order while upsampling the bottleneck fine→coarse.
        target_sizes = [
            (GRID_SIZE * 2,  GRID_SIZE * 2),   # 10→20
            (GRID_SIZE * 4,  GRID_SIZE * 4),   # 20→40  (approx; final crop fixes exact)
            (GRID_SIZE * 8,  GRID_SIZE * 8),   # 40→80
            (GRID_SIZE * 10, GRID_SIZE * 10),  # 80→100 → then crop to 128
        ]
        # Actually we just upsample by ×2 each stage and fix to H,W at the end.
        for i, conv in enumerate(self.decoder):
            z = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
            skip = self.skip_proj[i](skips[i])           # project to same ch as z

            # Resize skip to match z's current spatial size (handles any rounding)
            if skip.shape[-2:] != z.shape[-2:]:
                skip = F.interpolate(skip, size=z.shape[-2:],
                                     mode="bilinear", align_corners=False)

            z = conv(torch.cat([z, skip], dim=1))

        # --- output ----------------------------------------------------------
        out = self.out_conv(z)                           # (B, 1, ?, ?)

        # Resize to original input dimensions (handles the 140→128 offset)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return torch.sigmoid(out)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse_val = torch.mean((pred - target) ** 2).item()
    if mse_val == 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse_val)


@torch.no_grad()
def compute_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds, targets = [], []
    for noisy_b, clean_b in loader:
        pred = model(noisy_b.to(device)).cpu()
        preds.append(pred)
        targets.append(clean_b)
    preds   = torch.cat(preds)
    targets = torch.cat(targets)
    return {"mse": mse(preds, targets), "psnr": psnr(preds, targets)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, count = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for noisy_b, clean_b in loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            pred = model(noisy_b)
            loss = criterion(pred, clean_b)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * noisy_b.size(0)
            count += noisy_b.size(0)

    return total / count


def train(
    model:       nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:      int   = 40,
    lr:          float = 1e-4,
    weight_decay: float = 1e-4,
    loss_fn:     str   = "mse",
    output_dir:  Path  = Path("results/dinov2"),
    run_name:    str   = "dinov2_denoiser",
    device:      torch.device = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    # Only pass parameters that actually need gradients to the optimiser —
    # this is important when the encoder is frozen.
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if loss_fn == "l1":
        criterion = nn.L1Loss()
    elif loss_fn == "combined":
        _mse_fn, _l1_fn = nn.MSELoss(), nn.L1Loss()
        class _Combined(nn.Module):
            def forward(self, p, t):
                return 0.5 * _mse_fn(p, t) + 0.5 * _l1_fn(p, t)
        criterion = _Combined()
    else:
        criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tr  = _run_epoch(model, train_loader, criterion, device, optimizer)
        val = _run_epoch(model, val_loader,   criterion, device)
        train_losses.append(tr)
        val_losses.append(val)
        scheduler.step()

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{run_name}] {epoch:3d}/{epochs}  "
                  f"train {tr:.6f}  val {val:.6f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    if HAS_MATPLOTLIB:
        _save_loss_plot(train_losses, val_losses, epochs, run_name, output_dir)

    return {
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "best_val_loss":    best_val,
        "final_train_loss": train_losses[-1],
        "final_val_loss":   val_losses[-1],
    }


def _save_loss_plot(train_losses, val_losses, epochs, name, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(1, epochs + 1)
    ax.plot(x, train_losses, label="Train")
    ax.plot(x, val_losses,   label="Validation", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves — {name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir / 'loss_curves.png'}")


def _save_comparison(model, val_ds, device, output_dir: Path, run_name: str, n: int = 4):
    if not HAS_MATPLOTLIB:
        return
    model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    for i in range(n):
        noisy_t, clean_t = val_ds[i]
        with torch.no_grad():
            pred = model(noisy_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        for ax, img, lbl in zip(axes[i],
                                [noisy_t.squeeze(), clean_t.squeeze(), pred],
                                ["Noisy", "Clean", "Denoised"]):
            ax.imshow(img if isinstance(img, np.ndarray) else img.numpy(),
                      cmap="gray", vmin=0, vmax=1)
            ax.set_title(lbl if i == 0 else "")
            ax.axis("off")
    fig.suptitle(f"{run_name} — validation samples")
    plt.tight_layout()
    out = output_dir / "comparison_val.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train/evaluate a DINOv2-based image denoiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- data ---------------------------------------------------------------
    p.add_argument("--noisy",       required=True,  help=".npy file with noisy images")
    p.add_argument("--clean",       required=True,  help=".npy file with clean images")
    p.add_argument("--val-split",   type=float, default=0.15)
    p.add_argument("--seed",        type=int,   default=42)

    # ---- model --------------------------------------------------------------
    p.add_argument("--freeze-encoder",  action="store_true", default=True,
                   help="Freeze the DINOv2 encoder (default: True)")
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false",
                   help="Unfreeze the entire DINOv2 encoder")
    p.add_argument("--unfreeze-blocks", type=int, default=0,
                   help="Unfreeze the last N transformer blocks (used with --freeze-encoder)")

    # ---- training -----------------------------------------------------------
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--loss",         choices=["mse", "l1", "combined"], default="mse")
    p.add_argument("--num-workers",  type=int,   default=4)

    # ---- I/O ----------------------------------------------------------------
    p.add_argument("--output-dir",  default="results/dinov2")
    p.add_argument("--run-name",    default="dinov2_denoiser")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to a .pt checkpoint to load before training/eval")

    # ---- eval-only ----------------------------------------------------------
    p.add_argument("--eval-only",   action="store_true",
                   help="Skip training; evaluate the checkpoint and exit")

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)

    # ---- data ---------------------------------------------------------------
    train_loader, val_loader, train_ds, val_ds = make_loaders(
        noisy_path  = args.noisy,
        clean_path  = args.clean,
        val_split   = args.val_split,
        batch_size  = args.batch_size,
        seed        = args.seed,
        num_workers = args.num_workers,
    )
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ---- model --------------------------------------------------------------
    print("Loading DINOv2-large …")
    model = DINOv2Denoiser(
        freeze_encoder  = args.freeze_encoder,
        unfreeze_last_n = args.unfreeze_blocks,
    ).to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- train or eval ------------------------------------------------------
    if not args.eval_only:
        history = train(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            epochs       = args.epochs,
            lr           = args.lr,
            weight_decay = args.weight_decay,
            loss_fn      = args.loss,
            output_dir   = output_dir,
            run_name     = args.run_name,
            device       = device,
        )
        # Reload best checkpoint for final evaluation
        model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

    metrics = compute_metrics(model, val_loader, device)
    print(f"\nVal MSE:  {metrics['mse']:.6f}")
    print(f"Val PSNR: {metrics['psnr']:.2f} dB")

    _save_comparison(model, val_ds, device, output_dir, args.run_name)


if __name__ == "__main__":
    main()