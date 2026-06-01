"""
Training and evaluation script for the Conditional DDPM image denoiser.

Usage:
    # Quick test (few epochs):
    python src/diffusion_main.py --epochs 10

    # Default run (recommended):
    python src/diffusion_main.py --epochs 100 --loss l1

    # Best perceptual quality (slow, uses MS-SSIM+L1 loss):
    python src/diffusion_main.py --epochs 200 --loss mix

    # Larger backbone:
    python src/diffusion_main.py --epochs 100 --features 64 128 256

    # Custom experiment name:
    python src/diffusion_main.py --epochs 100 --loss mix --name ddpm_mix_loss
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import load_data, load_test                    # noqa: E402
from diffusion_model import CondDDPMUNet, GaussianDiffusion  # noqa: E402
from diffusion_trainer import fit_diffusion, validate        # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Conditional DDPM denoiser")
    p.add_argument("--epochs",     type=int,   default=50,
                   help="Training epochs (default: 50)")
    p.add_argument("--loss",       type=str,   default="l1",
                   choices=["l1", "mse", "mix"],
                   help="Training loss: l1 (default) | mse | mix (MS-SSIM+L1)")
    p.add_argument("--features",   type=int,   nargs=3, default=[32, 64, 128],
                   metavar=("F0", "F1", "F2"),
                   help="U-Net channel widths (default: 32 64 128)")
    p.add_argument("--T",          type=int,   default=1000,
                   help="Number of diffusion timesteps (default: 1000)")
    p.add_argument("--ddim-steps", type=int,   default=50,
                   help="DDIM inference steps for final evaluation (default: 50)")
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--t-dim",      type=int,   default=128,
                   help="Sinusoidal timestep embedding dimension (default: 128)")
    p.add_argument("--val-split",  type=float, default=0.05,
                   help="Fraction of data held out for validation (default: 0.05)")
    p.add_argument("--name",       type=str,   default="conditional_ddpm")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _save_comparison(diffusion_model, train_ds, val_ds, device, out_dir, run_name, n_ddim_steps):
    """Save side-by-side: sensor-noisy / clean ground truth / DDIM denoised."""
    def _triplet(ds, idx, label, fname):
        noisy_t, clean_t = ds[idx]
        diffusion_model.model.eval()
        with torch.no_grad():
            pred = diffusion_model.ddim_sample(
                noisy_t.unsqueeze(0).to(device), n_steps=n_ddim_steps
            ).squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for ax, img, lbl in zip(
            axes,
            [noisy_t.squeeze().numpy(), clean_t.squeeze().numpy(), pred],
            ["Noisy (sensor)", "Clean (target)",
             f"DDPM denoised\n({n_ddim_steps} DDIM steps)"],
        ):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(lbl)
            ax.axis("off")
        fig.suptitle(f"{run_name} — {label}")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    _triplet(train_ds, 0, "Training Sample",   "comparison_train.png")
    _triplet(val_ds,   0, "Validation Sample",  "comparison_val.png")


def _save_test_predictions(diffusion_model, test_ds, device, out_dir, n_ddim_steps, batch_size):
    """Run DDIM on the blind test set and save predictions."""
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    diffusion_model.model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            pred = diffusion_model.ddim_sample(batch.to(device), n_steps=n_ddim_steps)
            preds.append(pred.squeeze(1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)   # (N, H, W)
    np.save(out_dir / "test_predictions.npy", preds)

    # Quick visual grid
    n_show = min(8, len(test_ds))
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2, 4))
    for i in range(n_show):
        axes[0, i].imshow(test_ds[i].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(preds[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Noisy", fontsize=9)
    axes[1, 0].set_ylabel(f"DDIM\n({n_ddim_steps} steps)", fontsize=9)
    fig.suptitle(f"Blind test predictions — {run_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "test_predictions_grid.png", dpi=150)
    plt.close()
    print(f"  -> Saved test predictions ({len(preds)} images): "
          f"{out_dir / 'test_predictions.npy'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    data_cfg = {
        "noisy_path":      "data/noisy_train_19k_harder.npy",
        "clean_path":      "data/clean_train_19k_harder.npy",
        "test_noisy_path": "data/noisy_val_1k_harder.npy",
        "val_split":       args.val_split,
    }
    train_ds, val_ds = load_data(data_cfg, ROOT, seed=42)
    test_ds          = load_test(data_cfg, ROOT)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}"
          + (f"  Test: {len(test_ds)}" if test_ds else ""))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Build model
    unet      = CondDDPMUNet(features=args.features, t_dim=args.t_dim)
    diffusion = GaussianDiffusion(unet, T=args.T).to(device)
    n_params  = sum(p.numel() for p in diffusion.model.parameters() if p.requires_grad)
    print(f"Parameters (U-Net backbone): {n_params:,}")

    output_dir = ROOT / "results" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = {
        "epochs":        args.epochs,
        "lr":            args.lr,
        "weight_decay":  1e-4,
        "loss":          args.loss,
        "val_ddim_steps": 20,   # coarser DDIM during training for speed
        "val_every":     5,
    }

    print(f"\n{'=' * 60}")
    print(f"Conditional DDPM — {args.name}")
    print(f"{'=' * 60}")
    print(f"  T={args.T}  features={args.features}  loss={args.loss}")
    print(f"  DDIM steps at inference: {args.ddim_steps}")
    print(f"  Epochs: {args.epochs}")

    # Train
    history = fit_diffusion(
        diffusion, train_loader, val_loader,
        train_cfg, device, output_dir, args.name,
    )

    # Reload best checkpoint
    diffusion.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device)
    )

    # Final evaluation with full DDIM steps
    val_mse, val_psnr = validate(
        diffusion, val_loader, device, n_ddim_steps=args.ddim_steps
    )
    print(f"\nFinal evaluation ({args.ddim_steps} DDIM steps) — "
          f"Val MSE: {val_mse:.6f}  PSNR: {val_psnr:.2f} dB")

    # Qualitative comparison
    _save_comparison(diffusion, train_ds, val_ds, device, output_dir,
                     args.name, args.ddim_steps)

    # Blind test predictions
    if test_ds:
        _save_test_predictions(diffusion, test_ds, device, output_dir,
                               args.ddim_steps, args.batch_size)

    # Save metrics summary
    metrics = {
        "name":           args.name,
        "architecture":   "conditional_ddpm",
        "T":              args.T,
        "ddim_steps":     args.ddim_steps,
        "features":       args.features,
        "loss":           args.loss,
        "params":         n_params,
        "best_val_psnr":  history["best_val_psnr"],
        "best_val_mse":   history["best_val_mse"],
        "val_psnr_final": val_psnr,
        "val_mse_final":  val_mse,
        "train_losses":   history["train_losses"],
        "val_psnrs":      history["val_psnrs"],
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
