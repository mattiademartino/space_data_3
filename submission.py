"""
submission.py — Space Data Task 3: Lunar Image Denoising
=========================================================
Pretrained model: https://huggingface.co/mattiademartino/unet-deep-lunar-denoiser

Best model: unet_deep_01
Architecture: 4-stage UNet (UNetDeep) with features [64, 128, 256, 512]
Training: MSE loss, AdamW, cosine-annealing LR schedule, 150 epochs

Usage
-----
  # Train from scratch
  python submission.py train --data-dir data/ --out-dir results/submission/

  # Run inference on test set with local weights
  python submission.py test --data-dir data/ --weights best_model_mae.pt --out predictions.npy

  # Download pretrained weights from Hugging Face and run inference
  python submission.py test --data-dir data/ --hf-model mattiademartino/unet-deep-lunar-denoiser --out predictions.npy

  # Upload trained weights to Hugging Face
  python submission.py upload --weights best_model_mae.pt --hf-repo mattiademartino/unet-deep-lunar-denoiser
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Model
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


class UNetDeep(nn.Module):
    """4-stage U-Net; bottleneck at 8x8 for 128x128 inputs."""

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


def build_model() -> UNetDeep:
    """Return the unet_deep_01 architecture with its exact HPO hyperparameters."""
    return UNetDeep(features=[64, 128, 256, 512], dropout=0.10887)


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
    """Noisy-only dataset for blind inference."""

    def __init__(self, noisy: np.ndarray):
        self.noisy = torch.from_numpy(_to_float32(noisy)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.noisy[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

EPOCHS     = 150
VAL_SPLIT  = 0.05
BATCH_SIZE = 16
LR         = 9.664e-4
WEIGHT_DECAY = 2.472e-3


@torch.no_grad()
def _val_metrics(model: nn.Module, loader: DataLoader, device: torch.device):
    """Returns (mse, mae, psnr_dB) over the validation set."""
    model.eval()
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    mse_sum = mae_sum = n = 0.0
    for noisy_b, clean_b in loader:
        noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
        pred = model(noisy_b)
        bs = noisy_b.size(0)
        mse_sum += mse_fn(pred, clean_b).item() * bs
        mae_sum += mae_fn(pred, clean_b).item() * bs
        n += bs
    mse = mse_sum / n
    mae = mae_sum / n
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    return mse, mae, psnr


def train(data_dir: str | Path = "data", out_dir: str | Path = "results/submission") -> Path:
    """
    Train unet_deep_01 from scratch.

    Parameters
    ----------
    data_dir : path to the directory containing
               noisy_train_19k_harder.npy and clean_train_19k_harder.npy
    out_dir  : where to save best_model_mae.pt, best_model_mse.pt, final_model.pt

    Returns
    -------
    Path to best_model_mae.pt (best checkpoint by validation MAE)
    """
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    noisy = np.load(data_dir / "noisy_train_19k_harder.npy")
    clean = np.load(data_dir / "clean_train_19k_harder.npy")
    dataset = LunarDataset(noisy, clean)

    val_size   = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_mse = best_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
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
            print(
                f"  [{epoch:3d}/{EPOCHS}]  "
                f"train {tr_loss/tr_n:.6f}  "
                f"val_mse {val_mse:.6f}  val_mae {val_mae:.6f}  PSNR {val_psnr:.2f} dB"
            )

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    print(f"\nBest val MSE: {best_mse:.6f}  Best val MAE: {best_mae:.6f}")
    print(f"Weights saved to {out_dir}/")
    return out_dir / "best_model_mae.pt"


# ---------------------------------------------------------------------------
# Inference / test
# ---------------------------------------------------------------------------

def _load_weights(model: UNetDeep, weights_path: str | Path | None,
                  hf_model: str | None, device: torch.device) -> UNetDeep:
    """Load weights either from a local .pt file or from Hugging Face Hub."""
    if weights_path is not None:
        print(f"Loading local weights: {weights_path}")
        state = torch.load(weights_path, map_location=device)
    elif hf_model is not None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            sys.exit("Install huggingface_hub:  pip install huggingface_hub")
        print(f"Downloading weights from Hugging Face: {hf_model}")
        local_pt = hf_hub_download(repo_id=hf_model, filename="best_model_mae.pt")
        state = torch.load(local_pt, map_location=device)
    else:
        sys.exit("Provide either --weights or --hf-model")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model


def _save_preview_images(
    noisy: np.ndarray,
    predictions: np.ndarray,
    preview_dir: str | Path,
    num_preview: int,
) -> Path | None:
    """Save a few PNG previews for quick visual inspection."""
    if num_preview <= 0:
        return None

    import matplotlib.pyplot as plt

    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    noisy = _to_float32(noisy)
    n = min(num_preview, len(predictions))

    for idx in range(n):
        pred_u8 = (np.clip(predictions[idx], 0.0, 1.0) * 255).astype(np.uint8)
        plt.imsave(preview_dir / f"prediction_{idx:04d}.png", pred_u8, cmap="gray", vmin=0, vmax=255)

    grid_n = min(n, 8)
    if grid_n > 0:
        fig, axes = plt.subplots(2, grid_n, figsize=(2.0 * grid_n, 4.0))
        if grid_n == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        for idx in range(grid_n):
            axes[0, idx].imshow(noisy[idx], cmap="gray", vmin=0, vmax=1)
            axes[0, idx].set_title(f"Noisy {idx}")
            axes[1, idx].imshow(predictions[idx], cmap="gray", vmin=0, vmax=1)
            axes[1, idx].set_title(f"Pred {idx}")
        for ax in axes.ravel():
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(preview_dir / "preview_grid.png", dpi=160)
        plt.close(fig)

    return preview_dir.resolve()


def test(
    data_dir: str | Path = "data",
    weights_path: str | Path | None = "best_model_mae.pt",
    hf_model: str | None = None,
    out_path: str | Path = "predictions.npy",
    batch_size: int = 64,
    preview_dir: str | Path | None = None,
    num_preview: int = 16,
) -> np.ndarray:
    """
    Run inference on the blind test set.

    Parameters
    ----------
    data_dir      : directory containing noisy_val_500_harder.npy
    weights_path  : path to a local .pt checkpoint (takes priority over hf_model)
    hf_model      : Hugging Face repo id, e.g. 'demartinomattia/unet-deep-lunar-denoiser'
    out_path      : where to write the predictions array (N, 128, 128) float32 in [0, 1]
    batch_size    : inference batch size
    preview_dir   : where to write PNG previews
    num_preview   : number of prediction PNGs to save; set 0 to disable

    Returns
    -------
    numpy array of predictions, shape (N, 128, 128)
    """
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is None:
        preview_dir = out_path.with_suffix("").parent / f"{out_path.stem}_preview_images"

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model().to(device)
    model = _load_weights(model, weights_path, hf_model, device)
    model.eval()

    noisy_test = np.load(data_dir / "noisy_val_500_harder.npy")
    test_ds    = LunarTestDataset(noisy_test)
    loader     = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    preds = []
    with torch.no_grad():
        for noisy_b in loader:
            pred = model(noisy_b.to(device)).cpu().squeeze(1)
            preds.append(pred.numpy())

    predictions = np.concatenate(preds, axis=0)  # (N, 128, 128)
    np.save(out_path, predictions)

    noisy_float = _to_float32(noisy_test)
    mean_abs_delta = float(np.mean(np.abs(predictions - noisy_float)))
    max_abs_delta = float(np.max(np.abs(predictions - noisy_float)))
    saved_preview_dir = _save_preview_images(noisy_test, predictions, preview_dir, num_preview)

    print(
        f"Predictions array saved to {out_path.resolve()}  shape={predictions.shape}  "
        f"mean|pred-noisy|={mean_abs_delta:.6f}  max|pred-noisy|={max_abs_delta:.6f}"
    )
    if saved_preview_dir is not None:
        print(f"Preview images saved to {saved_preview_dir}")
    return predictions


# ---------------------------------------------------------------------------
# Hugging Face upload
# ---------------------------------------------------------------------------

def upload_to_hf(weights_path: str | Path, hf_repo: str) -> None:
    """
    Upload a trained .pt checkpoint to a Hugging Face model repository.

    Parameters
    ----------
    weights_path : local path to the .pt file
    hf_repo      : Hugging Face repo id, e.g. 'demartinomattia/unet-deep-lunar-denoiser'
                   The repo is created automatically if it does not exist.
    """
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
        commit_message=f"Upload {weights_path.name} (unet_deep_01)",
    )
    print(f"Uploaded {weights_path} → https://huggingface.co/{hf_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lunar denoising — unet_deep_01 submission script",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train unet_deep_01 from scratch")
    p_train.add_argument("--data-dir", default="data",
                         help="Directory with noisy/clean .npy files (default: data/)")
    p_train.add_argument("--out-dir",  default="results/submission",
                         help="Output directory for checkpoints (default: results/submission/)")

    # --- test ---
    p_test = sub.add_parser("test", help="Run inference on the test set")
    p_test.add_argument("--data-dir",  default="data",
                        help="Directory with noisy_val_500_harder.npy (default: data/)")
    p_test.add_argument("--weights",   default=None,
                        help="Path to a local .pt checkpoint (overrides --hf-model)")
    p_test.add_argument("--hf-model",  default=None,
                        help="Hugging Face repo id, e.g. mattiademartino/unet-deep-lunar-denoiser")
    p_test.add_argument("--out",       default="predictions.npy",
                        help="Output .npy file path (default: predictions.npy)")
    p_test.add_argument("--batch-size", type=int, default=64)
    p_test.add_argument("--preview-dir", default=None,
                        help="Directory for PNG preview images (default: <out>_preview_images)")
    p_test.add_argument("--num-preview", type=int, default=16,
                        help="Number of PNG prediction previews to save; use 0 to disable")

    # --- upload ---
    p_up = sub.add_parser("upload", help="Upload weights to Hugging Face Hub")
    p_up.add_argument("--weights",  required=True, help="Path to the .pt file to upload")
    p_up.add_argument("--hf-repo",  required=True,
                      help="Hugging Face repo id (e.g. demartinomattia/unet-deep-lunar-denoiser)")

    args = parser.parse_args()

    if args.cmd == "train":
        train(data_dir=args.data_dir, out_dir=args.out_dir)

    elif args.cmd == "test":
        test(
            data_dir=args.data_dir,
            weights_path=args.weights,
            hf_model=args.hf_model,
            out_path=args.out,
            batch_size=args.batch_size,
            preview_dir=args.preview_dir,
            num_preview=args.num_preview,
        )

    elif args.cmd == "upload":
        upload_to_hf(weights_path=args.weights, hf_repo=args.hf_repo)


if __name__ == "__main__":
    main()
