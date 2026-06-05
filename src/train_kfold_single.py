"""
Train a single fold of k-fold cross-validation for unet_deep_01.

This script is meant to be called once per fold by submit_kfold.sh.
Results are saved to results/kfold_unet_deep_01/fold_<k>/.

Usage:
    python src/train_kfold_single.py --fold 0 --n-folds 5
    python src/train_kfold_single.py --fold 1 --n-folds 5 --epochs 150
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import load_data    # noqa: E402
from models import build_model   # noqa: E402
from trainer import build_criterion  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed model spec from Optuna best trial
# ---------------------------------------------------------------------------

MODEL_CFG = {
    "architecture": "unet_deep",
    "features":     [64, 128, 256, 512],
    "dropout":      0.10887,
    "output_fct":   "sigmoid",
}

TRAIN_CFG = {
    "loss":         "mse",
    "lr":           0.0009664,
    "weight_decay": 0.002472,
    "optimizer":    "adamw",
    "batch_size":   16,
}

# ---------------------------------------------------------------------------
# PSNR helper
# ---------------------------------------------------------------------------

_mse_fn = nn.MSELoss()


def _val_psnr(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for noisy_b, clean_b in loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            mse = _mse_fn(model(noisy_b), clean_b)
            total += mse.item() * noisy_b.size(0)
            count += noisy_b.size(0)
    mse_val = total / count
    return 10.0 * math.log10(1.0 / mse_val) if mse_val > 0 else float("inf")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total, count = 0.0, 0
    for noisy_b, clean_b in loader:
        noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
        loss = criterion(model(noisy_b), clean_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * noisy_b.size(0)
        count += noisy_b.size(0)
    return total / count

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train one k-fold split of unet_deep_01")
    p.add_argument("--fold",    type=int, required=True, help="0-based fold index")
    p.add_argument("--n-folds", type=int, default=5,     help="Total number of folds (default: 5)")
    p.add_argument("--epochs",  type=int, default=150,   help="Training epochs (default: 150)")
    p.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Fraction of each fold used for validation (85-15 split, default: 0.15)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    fold     = args.fold
    n_folds  = args.n_folds
    n_epochs = args.epochs

    assert 0 <= fold < n_folds, f"--fold must be in [0, {n_folds - 1}]"

    out_dir = ROOT / "results" / "kfold_unet_deep_01" / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42 + fold)   # different seed per fold for reproducibility
    np.random.seed(42 + fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fold {fold}/{n_folds - 1}] device={device}  epochs={n_epochs}  out={out_dir}")

    # Load full dataset (no pre-split; we do the splitting here via KFold)
    data_cfg = {
        "noisy_path": "data/noisy_train_19k_harder.npy",
        "clean_path": "data/clean_train_19k_harder.npy",
        "val_split":  0.0,   # load everything as "train"
    }
    full_ds, _ = load_data(data_cfg, ROOT, seed=42)
    n_total = len(full_ds)
    print(f"[fold {fold}] Full dataset size: {n_total}")

    # K-Fold split — 85 % train / 15 % val per fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(range(n_total)))
    train_idx, val_idx = splits[fold]

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    print(f"[fold {fold}] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model     = build_model(MODEL_CFG).to(device)
    criterion = build_criterion(TRAIN_CFG)

    lr = float(TRAIN_CFG["lr"])
    wd = float(TRAIN_CFG["weight_decay"])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = []
    best_psnr  = -float("inf")
    best_epoch = -1
    best_path  = out_dir / "best_model.pt"

    for epoch in range(1, n_epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_psnr = _val_psnr(model, val_loader, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_psnr": val_psnr})

        if val_psnr > best_psnr:
            best_psnr  = val_psnr
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == n_epochs:
            print(
                f"[fold {fold}] epoch {epoch:3d}/{n_epochs} "
                f"| train_loss={train_loss:.6f} "
                f"| val_psnr={val_psnr:.4f} dB "
                f"| best={best_psnr:.4f} dB (epoch {best_epoch})"
            )

    # Save fold summary
    summary = {
        "fold":        fold,
        "n_folds":     n_folds,
        "best_epoch":  best_epoch,
        "best_psnr":   best_psnr,
        "n_train":     len(train_ds),
        "n_val":       len(val_ds),
        "model_cfg":   MODEL_CFG,
        "train_cfg":   TRAIN_CFG,
        "history":     history,
    }
    with open(out_dir / "fold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[fold {fold}] DONE — best val PSNR = {best_psnr:.4f} dB at epoch {best_epoch}")
    print(f"[fold {fold}] Model saved to {best_path}")
    print(f"[fold {fold}] Summary saved to {out_dir / 'fold_summary.json'}")


if __name__ == "__main__":
    main()