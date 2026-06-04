"""
Full training run for the 10 best models selected from Optuna HPO.

Traccia ogni epoch: train loss, val MSE, val MAE, val PSNR.
Salva:
  best_model_mse.pt  → checkpoint con miglior val MSE
  best_model_mae.pt  → checkpoint con miglior val MAE
  final_model.pt     → ultimo epoch
  plot_mse.png       → train loss + val MSE per epoch
  plot_mae.png       → train loss + val MAE per epoch
  metrics.json       → tutte le metriche per epoch

Usage:
    python src/train_final.py --model-id 1
    python src/train_final.py --summarize
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import load_data                                          # noqa: E402
from models import build_model                                         # noqa: E402
from trainer import build_criterion, build_optimizer, build_scheduler  # noqa: E402

# ---------------------------------------------------------------------------
# Feature map presets
# ---------------------------------------------------------------------------

_FM4 = {"small": [16, 32, 64, 128], "default": [32, 64, 128, 256], "large": [64, 128, 256, 512]}
_FM3 = {"small": [16, 32, 64],      "default": [32, 64, 128],       "large": [64, 128, 256]}

# ---------------------------------------------------------------------------
# 10 model configs
# ---------------------------------------------------------------------------

MODELS = {
    1: {
        "name": "unet_deep_01",
        "arch": "unet_deep",
        "model_cfg": {"architecture": "unet_deep", "features": _FM4["large"],
                      "dropout": 0.10887, "output_fct": "sigmoid"},
        "train_cfg": {"loss": "mse", "lr": 9.664e-4, "weight_decay": 2.472e-3,
                      "optimizer": "adamw", "batch_size": 16},
        "optuna_psnr": 27.6876,
    },
    2: {
        "name": "unet_deep_02",
        "arch": "unet_deep",
        "model_cfg": {"architecture": "unet_deep", "features": _FM4["large"],
                      "dropout": 0.17642, "output_fct": "sigmoid"},
        "train_cfg": {"loss": "mse", "lr": 4.622e-3, "weight_decay": 2.593e-3,
                      "optimizer": "adamw", "batch_size": 16},
        "optuna_psnr": 27.6781,
    },
    3: {
        "name": "unet_deep_03",
        "arch": "unet_deep",
        "model_cfg": {"architecture": "unet_deep", "features": _FM4["large"],
                      "dropout": 0.13229, "output_fct": "sigmoid"},
        "train_cfg": {"loss": "mse", "lr": 1.776e-3, "weight_decay": 2.491e-3,
                      "optimizer": "adamw", "batch_size": 16},
        "optuna_psnr": 27.6652,
    },
    4: {
        "name": "unet_attention_01",
        "arch": "unet_attention",
        "model_cfg": {"architecture": "unet_attention", "features": _FM3["large"],
                      "dropout": 0.02709, "output_fct": "sigmoid"},
        "train_cfg": {"loss": "combined", "lr": 2.578e-3, "weight_decay": 1.490e-4,
                      "optimizer": "adamw", "batch_size": 16},
        "optuna_psnr": 26.2096,
    },
    5: {
        "name": "unet_attention_02",
        "arch": "unet_attention",
        "model_cfg": {"architecture": "unet_attention", "features": _FM3["default"],
                      "dropout": 0.05561, "output_fct": "sigmoid"},
        "train_cfg": {"loss": "combined", "lr": 4.007e-4, "weight_decay": 1.687e-4,
                      "optimizer": "adamw", "batch_size": 64},
        "optuna_psnr": 26.0374,
    },
    6: {
        "name": "unet_attention_03",
        "arch": "unet_attention",
        "model_cfg": {"architecture": "unet_attention", "features": _FM3["default"],
                      "dropout": 0.04915, "output_fct": "sigmoid"},
        "train_cfg": {"loss": "combined", "lr": 3.867e-4, "weight_decay": 2.013e-4,
                      "optimizer": "adamw", "batch_size": 64},
        "optuna_psnr": 26.0237,
    },
    7: {
        "name": "resnet_01",
        "arch": "resnet",
        "model_cfg": {"architecture": "resnet", "base_channels": 32, "n_blocks": 16,
                      "dropout": 0.27032, "output_fct": "sigmoid"},
        "train_cfg": {"loss": "combined", "lr": 5.401e-4, "weight_decay": 5.993e-5,
                      "optimizer": "adam", "batch_size": 16},
        "optuna_psnr": 25.4966,
    },
    8: {
        "name": "resnet_02",
        "arch": "resnet",
        "model_cfg": {"architecture": "resnet", "base_channels": 64, "n_blocks": 16,
                      "dropout": 0.25537, "output_fct": "tanh"},
        "train_cfg": {"loss": "combined", "lr": 2.021e-4, "weight_decay": 2.346e-3,
                      "optimizer": "adamw", "batch_size": 32},
        "optuna_psnr": 25.4527,
    },
    9: {
        "name": "diffusion_01",
        "arch": "diffusion",
        "model_cfg": {"features": _FM3["large"], "t_dim": 128, "T": 1000},
        "train_cfg": {"loss": "mse", "lr": 4.248e-4, "weight_decay": 4.583e-5,
                      "optimizer": "adamw", "batch_size": 32},
        "optuna_psnr": 25.5303,
    },
    10: {
        "name": "diffusion_02",
        "arch": "diffusion",
        "model_cfg": {"features": _FM3["large"], "t_dim": 128, "T": 1000},
        "train_cfg": {"loss": "mse", "lr": 3.540e-4, "weight_decay": 1.068e-5,
                      "optimizer": "adamw", "batch_size": 32},
        "optuna_psnr": 25.5241,
    },
}

FULL_EPOCHS    = 150
VAL_SPLIT      = 0.05
VAL_DDIM_STEPS = 20

# ---------------------------------------------------------------------------
# Val metrics: MSE, MAE, PSNR — computed in one pass
# ---------------------------------------------------------------------------

_mse_fn = nn.MSELoss()
_mae_fn = nn.L1Loss()


@torch.no_grad()
def compute_val_metrics(model, loader, device) -> tuple[float, float, float]:
    """Returns (val_mse, val_mae, val_psnr)."""
    model.eval()
    mse_sum = mae_sum = count = 0.0
    for noisy_b, clean_b in loader:
        noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
        pred = model(noisy_b)
        n = noisy_b.size(0)
        mse_sum += _mse_fn(pred, clean_b).item() * n
        mae_sum += _mae_fn(pred, clean_b).item() * n
        count   += n
    mse  = mse_sum / count
    mae  = mae_sum / count
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    return mse, mae, psnr

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plot(epochs, train_losses, val_values, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_values,   label=f"Val {ylabel}", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plots(train_losses, val_mse, val_mae, name, out_dir, val_epochs=None):
    """val_epochs: list of epoch indices (for sparse diffusion validation)."""
    ep = val_epochs if val_epochs is not None else list(range(1, len(train_losses) + 1))
    tr = train_losses if val_epochs is None else [train_losses[e - 1] for e in ep]

    _make_plot(ep, tr, val_mse, "MSE",
               f"Val MSE — {name}", out_dir / "plot_mse.png")
    _make_plot(ep, tr, val_mae, "MAE",
               f"Val MAE — {name}", out_dir / "plot_mae.png")

# ---------------------------------------------------------------------------
# Feed-forward training
# ---------------------------------------------------------------------------

def _save_checkpoint(out_dir, epoch, model, optimizer, scheduler,
                     best_mse, best_mae, train_losses, val_mse_list, val_mae_list, val_psnr_list):
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_mse":       best_mse,
        "best_mae":       best_mae,
        "train_losses":   train_losses,
        "val_mse_list":   val_mse_list,
        "val_mae_list":   val_mae_list,
        "val_psnr_list":  val_psnr_list,
    }, out_dir / "checkpoint.pt")


def train_feedforward(cfg, out_dir, device, train_ds, val_ds) -> dict:
    name       = cfg["name"]
    train_cfg  = {**cfg["train_cfg"], "epochs": FULL_EPOCHS, "scheduler": "cosine"}
    batch_size = int(train_cfg["batch_size"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = build_model(cfg["model_cfg"]).to(device)
    criterion = build_criterion(train_cfg)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, FULL_EPOCHS)

    train_losses  = []
    val_mse_list  = []
    val_mae_list  = []
    val_psnr_list = []
    best_mse = float("inf")
    best_mae = float("inf")
    start_epoch = 1

    # Resume from checkpoint if present
    ckpt_path = out_dir / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt["scheduler_state"]:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_mse      = ckpt["best_mse"]
        best_mae      = ckpt["best_mae"]
        train_losses  = ckpt["train_losses"]
        val_mse_list  = ckpt["val_mse_list"]
        val_mae_list  = ckpt["val_mae_list"]
        val_psnr_list = ckpt["val_psnr_list"]
        print(f"  Resumed from epoch {ckpt['epoch']}  best_mse={best_mse:.6f}")

    for epoch in range(start_epoch, FULL_EPOCHS + 1):
        model.train()
        tr_total, tr_count = 0.0, 0
        for noisy_b, clean_b in train_loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            loss = criterion(model(noisy_b), clean_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_total += loss.item() * noisy_b.size(0)
            tr_count += noisy_b.size(0)

        if scheduler:
            scheduler.step()

        val_mse, val_mae, val_psnr = compute_val_metrics(model, val_loader, device)

        train_losses.append(tr_total / tr_count)
        val_mse_list.append(val_mse)
        val_mae_list.append(val_mae)
        val_psnr_list.append(val_psnr)

        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), out_dir / "best_model_mse.pt")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), out_dir / "best_model_mae.pt")

        # Save full checkpoint every 10 epochs for resume
        if epoch % 10 == 0:
            _save_checkpoint(out_dir, epoch, model, optimizer, scheduler,
                             best_mse, best_mae, train_losses,
                             val_mse_list, val_mae_list, val_psnr_list)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{name}] {epoch:3d}/{FULL_EPOCHS}  "
                  f"train {train_losses[-1]:.6f}  "
                  f"val_mse {val_mse:.6f}  val_mae {val_mae:.6f}  PSNR {val_psnr:.2f} dB")

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    save_plots(train_losses, val_mse_list, val_mae_list, name, out_dir)

    return {
        "train_losses": train_losses,
        "val_mse":      val_mse_list,
        "val_mae":      val_mae_list,
        "val_psnr_dB":  val_psnr_list,
        "best_val_mse": best_mse,
        "best_val_mae": best_mae,
        "best_psnr_dB": max(val_psnr_list),
    }

# ---------------------------------------------------------------------------
# Diffusion training
# ---------------------------------------------------------------------------

def train_diffusion(cfg, out_dir, device, train_ds, val_ds) -> dict:
    from diffusion_model   import CondDDPMUNet, GaussianDiffusion  # noqa: E402
    from diffusion_trainer import validate as ddim_validate          # noqa: E402

    name       = cfg["name"]
    train_cfg  = cfg["train_cfg"]
    model_cfg  = cfg["model_cfg"]
    batch_size = int(train_cfg["batch_size"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    unet      = CondDDPMUNet(features=model_cfg["features"], t_dim=model_cfg["t_dim"])
    diffusion = GaussianDiffusion(unet, T=model_cfg["T"]).to(device)
    mse_crit  = nn.MSELoss()

    lr, wd = float(train_cfg["lr"]), float(train_cfg["weight_decay"])
    optimizer = (optim.Adam if train_cfg["optimizer"] == "adam" else optim.AdamW)(
        diffusion.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FULL_EPOCHS)

    val_every = max(1, FULL_EPOCHS // 15)

    train_losses  = []
    val_epochs    = []
    val_mse_list  = []
    val_mae_list  = []
    val_psnr_list = []
    best_mse = float("inf")
    best_mae = float("inf")

    for epoch in range(1, FULL_EPOCHS + 1):
        diffusion.model.train()
        ep_loss, n_b = 0.0, 0
        for noisy_b, clean_b in train_loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            loss = diffusion.training_step(clean_b, noisy_b, mse_crit)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_b     += 1
        scheduler.step()
        train_losses.append(ep_loss / n_b)

        if (epoch % val_every == 0) or (epoch == FULL_EPOCHS):
            # ddim_validate returns (mse, psnr); compute MAE manually
            val_mse, val_psnr = ddim_validate(diffusion, val_loader, device, VAL_DDIM_STEPS)

            # MAE from DDIM predictions
            diffusion.model.eval()
            mae_sum, count = 0.0, 0
            with torch.no_grad():
                for noisy_b, clean_b in val_loader:
                    noisy_b = noisy_b.to(device)
                    pred    = diffusion.ddim_sample(noisy_b, n_steps=VAL_DDIM_STEPS).cpu()
                    mae_sum += _mae_fn(pred, clean_b).item() * noisy_b.size(0)
                    count   += noisy_b.size(0)
            val_mae = mae_sum / count

            val_epochs.append(epoch)
            val_mse_list.append(val_mse)
            val_mae_list.append(val_mae)
            val_psnr_list.append(val_psnr)

            if val_mse < best_mse:
                best_mse = val_mse
                torch.save(diffusion.state_dict(), out_dir / "best_model_mse.pt")

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(diffusion.state_dict(), out_dir / "best_model_mae.pt")

            print(f"  [{name}] {epoch:3d}/{FULL_EPOCHS}  "
                  f"train {train_losses[-1]:.5f}  "
                  f"val_mse {val_mse:.6f}  val_mae {val_mae:.6f}  PSNR {val_psnr:.2f} dB")

    torch.save(diffusion.state_dict(), out_dir / "final_model.pt")
    save_plots(train_losses, val_mse_list, val_mae_list, name, out_dir,
               val_epochs=val_epochs)

    return {
        "val_epochs":   val_epochs,
        "train_losses": train_losses,
        "val_mse":      val_mse_list,
        "val_mae":      val_mae_list,
        "val_psnr_dB":  val_psnr_list,
        "best_val_mse": best_mse,
        "best_val_mae": best_mae,
        "best_psnr_dB": max(val_psnr_list) if val_psnr_list else None,
    }

# ---------------------------------------------------------------------------
# Entry point per singolo modello
# ---------------------------------------------------------------------------

def train_model(model_id: int):
    cfg     = MODELS[model_id]
    name    = cfg["name"]
    out_dir = ROOT / "results" / "final" / f"model_{model_id:02d}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model {model_id:2d}/10 — {name}")
    print(f"  Arch       : {cfg['arch']}")
    print(f"  Loss       : {cfg['train_cfg']['loss']}")
    print(f"  Epochs     : {FULL_EPOCHS}")
    print(f"  Optuna PSNR: {cfg['optuna_psnr']:.4f} dB")
    print(f"  Output     : {out_dir}")
    print(f"{'='*60}\n")

    with open(out_dir / "config.json", "w") as f:
        json.dump({"model_id": model_id, "name": name, "arch": cfg["arch"],
                   "epochs": FULL_EPOCHS, "val_split": VAL_SPLIT,
                   "optuna_psnr": cfg["optuna_psnr"],
                   "model_cfg": cfg["model_cfg"],
                   "train_cfg": cfg["train_cfg"]}, f, indent=2)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = {"noisy_path": "data/noisy_train_19k_harder.npy",
                "clean_path": "data/clean_train_19k_harder.npy",
                "val_split":  VAL_SPLIT}
    train_ds, val_ds = load_data(data_cfg, ROOT, seed=42)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}\n")

    if cfg["arch"] == "diffusion":
        results = train_diffusion(cfg, out_dir, device, train_ds, val_ds)
    else:
        results = train_feedforward(cfg, out_dir, device, train_ds, val_ds)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Best val MSE  : {results['best_val_mse']:.6f}")
    print(f"  Best val MAE  : {results['best_val_mae']:.6f}")
    print(f"  Best PSNR     : {results['best_psnr_dB']:.4f} dB")
    print(f"  Saved to      : {out_dir}\n")

# ---------------------------------------------------------------------------
# Summary cross-model
# ---------------------------------------------------------------------------

def summarize():
    import csv
    final_dir = ROOT / "results" / "final"
    rows = []
    for mid, cfg in MODELS.items():
        metrics_path = final_dir / f"model_{mid:02d}_{cfg['name']}" / "metrics.json"
        if not metrics_path.exists():
            print(f"  model_{mid:02d} — not yet trained")
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        rows.append({
            "id":          mid,
            "name":        cfg["name"],
            "arch":        cfg["arch"],
            "loss":        cfg["train_cfg"]["loss"],
            "optuna_psnr": cfg["optuna_psnr"],
            "best_val_mse": m["best_val_mse"],
            "best_val_mae": m["best_val_mae"],
            "best_psnr_dB": m["best_psnr_dB"],
        })

    if not rows:
        print("Nessun modello ancora addestrato.")
        return

    for metric, key, reverse in [("PSNR", "best_psnr_dB", True),
                                  ("MSE",  "best_val_mse",  False),
                                  ("MAE",  "best_val_mae",  False)]:
        ranked = sorted(rows, key=lambda r: -r[key] if reverse else r[key])
        print(f"\nRanking by {metric}:")
        print(f"  {'#':>3}  {'Name':<22}  {metric:>10}  PSNR (dB)")
        for rank, r in enumerate(ranked, 1):
            print(f"  #{rank:<2}  {r['name']:<22}  {r[key]:>10.6f}  {r['best_psnr_dB']:>8.4f}")

    csv_path = final_dir / "summary.csv"
    rows_psnr = sorted(rows, key=lambda r: -r["best_psnr_dB"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows_psnr)
    print(f"\nSalvato → {csv_path}")

    arch_colors = {"unet_deep": "steelblue", "unet_attention": "seagreen",
                   "resnet": "darkorange", "diffusion": "tomato"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, sort_key, reverse, ylabel, title in [
        (axes[0], "best_psnr_dB", True,  "PSNR (dB)", f"Best Val PSNR ({FULL_EPOCHS} ep)"),
        (axes[1], "best_val_mse", False, "MSE",        f"Best Val MSE ({FULL_EPOCHS} ep)"),
        (axes[2], "best_val_mae", False, "MAE",        f"Best Val MAE ({FULL_EPOCHS} ep)"),
    ]:
        ranked = sorted(rows, key=lambda r: -r[sort_key] if reverse else r[sort_key])
        names  = [r["name"] for r in ranked]
        vals   = [r[sort_key] for r in ranked]
        colors = [arch_colors.get(r["arch"], "gray") for r in ranked]
        bars   = ax.bar(range(len(names)), vals, color=colors, alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor=c, label=a, alpha=0.8)
                             for a, c in arch_colors.items()],
                   loc="lower right", fontsize=8)
    plt.suptitle("Final training — 10 best models from Optuna HPO", fontsize=11)
    plt.tight_layout()
    out_plot = final_dir / "summary.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Salvato → {out_plot}")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model-id",  type=int, choices=range(1, 11))
    g.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.summarize:
        summarize()
    else:
        train_model(args.model_id)


if __name__ == "__main__":
    main()
