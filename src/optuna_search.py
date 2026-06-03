"""
Optuna hyperparameter optimisation for four denoising models.

Each trial trains the model for --epochs-per-trial epochs and reports
validation PSNR (always computed via MSE so trials are comparable).
MedianPruner kills trials that fall below the median PSNR after the
warmup period, saving compute on clearly bad configurations.

Supported architectures (loss is FIXED, not tuned):
    unet_deep       → MSE loss
    resnet_16blocks → combined loss (0.5*MSE + 0.5*L1), n_blocks=16 fixed
    unet_attention  → combined loss
    diffusion       → MSE loss (conditional DDPM, NOT nvidia/mix)

Usage (on Euler — each arch is a separate job):
    python src/optuna_search.py --arch unet_deep
    python src/optuna_search.py --arch resnet_16blocks
    python src/optuna_search.py --arch unet_attention
    python src/optuna_search.py --arch diffusion

Override budget:
    python src/optuna_search.py --arch unet_deep --n-trials 60 --epochs-per-trial 50

Resume an interrupted study (same command, automatically continues):
    python src/optuna_search.py --arch unet_deep

Results are saved in results/optuna_<arch>/:
    best_params.json          best trial hyperparameters
    trials_summary.csv        all trial results sorted by PSNR
    optimization_history.png  PSNR improvement over trials
    param_importances.png     which hyperparameters matter most
    study.db                  SQLite database (allows resuming)
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import load_data   # noqa: E402
from models import build_model  # noqa: E402
from trainer import build_criterion  # noqa: E402

# ---------------------------------------------------------------------------
# Per-model default compute budgets  (n_trials, epochs_per_trial)
# ---------------------------------------------------------------------------

DEFAULT_BUDGET = {
    "unet_deep":       (40, 40),
    "resnet_16blocks": (40, 40),
    "unet_attention":  (40, 40),
    "diffusion":       (15, 30),   # DDIM validation is slow → lighter budget
}

# ---------------------------------------------------------------------------
# Feature-map presets
# ---------------------------------------------------------------------------

FEATURES_MAP = {                         # 4-stage (unet_deep)
    "small":   [16,  32,  64,  128],
    "default": [32,  64,  128, 256],
    "large":   [64,  128, 256, 512],
}

FEATURES_MAP_3 = {                       # 3-stage (unet_attention, diffusion)
    "small":   [16,  32,  64],
    "default": [32,  64,  128],
    "large":   [64,  128, 256],
}

# ---------------------------------------------------------------------------
# Search space definitions (loss is FIXED per model, never suggested)
# ---------------------------------------------------------------------------

def _suggest_unet_deep(trial: optuna.Trial) -> tuple[dict, dict]:
    features_key = trial.suggest_categorical("features", ["small", "default", "large"])
    model_cfg = {
        "architecture": "unet_deep",
        "features":     FEATURES_MAP[features_key],
        "dropout":      trial.suggest_float("dropout", 0.0, 0.3),
        "output_fct":   trial.suggest_categorical("output_fct", ["sigmoid", "tanh"]),
    }
    train_cfg = {
        "loss":         "mse",
        "lr":           trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "optimizer":    trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    return model_cfg, train_cfg


def _suggest_resnet16(trial: optuna.Trial) -> tuple[dict, dict]:
    model_cfg = {
        "architecture":  "resnet",
        "base_channels": trial.suggest_categorical("base_channels", [32, 64, 96, 128]),
        "n_blocks":      16,   # fixed
        "dropout":       trial.suggest_float("dropout", 0.0, 0.3),
        "output_fct":    trial.suggest_categorical("output_fct", ["sigmoid", "tanh"]),
    }
    train_cfg = {
        "loss":         "combined",
        "lr":           trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "optimizer":    trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    return model_cfg, train_cfg


def _suggest_unet_attention(trial: optuna.Trial) -> tuple[dict, dict]:
    features_key = trial.suggest_categorical("features", ["small", "default", "large"])
    model_cfg = {
        "architecture": "unet_attention",
        "features":     FEATURES_MAP_3[features_key],
        "dropout":      trial.suggest_float("dropout", 0.0, 0.3),
        "output_fct":   trial.suggest_categorical("output_fct", ["sigmoid", "tanh"]),
    }
    train_cfg = {
        "loss":         "combined",
        "lr":           trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "optimizer":    trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    return model_cfg, train_cfg


def _suggest_diffusion(trial: optuna.Trial) -> tuple[dict, dict]:
    features_key = trial.suggest_categorical("features", ["small", "default", "large"])
    model_cfg = {
        "features": FEATURES_MAP_3[features_key],
        "t_dim":    trial.suggest_categorical("t_dim", [64, 128, 256]),
        "T":        trial.suggest_categorical("T", [500, 1000]),
    }
    train_cfg = {
        "loss":         "mse",
        "lr":           trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "optimizer":    trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    return model_cfg, train_cfg


SUGGESTERS = {
    "unet_deep":       _suggest_unet_deep,
    "resnet_16blocks": _suggest_resnet16,
    "unet_attention":  _suggest_unet_attention,
}

# ---------------------------------------------------------------------------
# Shared evaluation helper: val PSNR via MSE (same metric for all models)
# ---------------------------------------------------------------------------

_mse_fn = nn.MSELoss()


def _val_psnr(model, loader, device) -> float:
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


def _run_train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for noisy_b, clean_b in loader:
        noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
        pred = model(noisy_b)
        loss = criterion(pred, clean_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * noisy_b.size(0)
        count += noisy_b.size(0)
    return total / count

# ---------------------------------------------------------------------------
# Objective for feed-forward models (unet_deep, resnet_16blocks, unet_attention)
# ---------------------------------------------------------------------------

def make_objective(train_ds, val_ds, arch: str, n_epochs: int, device: torch.device):
    suggest = SUGGESTERS[arch]

    def objective(trial: optuna.Trial) -> float:
        model_cfg, train_cfg = suggest(trial)
        batch_size = int(train_cfg["batch_size"])

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        model     = build_model(model_cfg).to(device)
        criterion = build_criterion(train_cfg)

        lr, wd = float(train_cfg["lr"]), float(train_cfg["weight_decay"])
        if train_cfg["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        best_psnr = -float("inf")

        try:
            for epoch in range(1, n_epochs + 1):
                _run_train_epoch(model, train_loader, criterion, optimizer, device)
                scheduler.step()

                psnr = _val_psnr(model, val_loader, device)
                best_psnr = max(best_psnr, psnr)

                trial.report(psnr, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            raise

        return best_psnr

    return objective

# ---------------------------------------------------------------------------
# Objective for diffusion model (DDIM validation, sparser reports)
# ---------------------------------------------------------------------------

def make_diffusion_objective(
    train_ds, val_ds, n_epochs: int, device: torch.device, val_ddim_steps: int
):
    from diffusion_model import CondDDPMUNet, GaussianDiffusion  # lazy import
    from diffusion_trainer import validate as ddim_validate

    criterion_mse = nn.MSELoss()
    val_every = max(1, n_epochs // 10)   # validate ~10 times per trial

    def objective(trial: optuna.Trial) -> float:
        model_cfg, train_cfg = _suggest_diffusion(trial)
        batch_size = int(train_cfg["batch_size"])

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        unet      = CondDDPMUNet(features=model_cfg["features"], t_dim=model_cfg["t_dim"])
        diffusion = GaussianDiffusion(unet, T=model_cfg["T"]).to(device)

        lr, wd = float(train_cfg["lr"]), float(train_cfg["weight_decay"])
        if train_cfg["optimizer"] == "adam":
            optimizer = optim.Adam(diffusion.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=wd)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        best_psnr = -float("inf")

        try:
            for epoch in range(1, n_epochs + 1):
                # Training step
                diffusion.model.train()
                for noisy_b, clean_b in train_loader:
                    noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
                    # training_step(x0, y, criterion): clean=x0, noisy=y
                    loss = diffusion.training_step(clean_b, noisy_b, criterion_mse)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                # Validate periodically and on last epoch
                do_val = (epoch % val_every == 0) or (epoch == n_epochs)
                if do_val:
                    _, psnr = ddim_validate(diffusion, val_loader, device, val_ddim_steps)
                    best_psnr = max(best_psnr, psnr)
                    trial.report(psnr, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            raise

        return best_psnr

    return objective

# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def _save_results(study: optuna.Study, out_dir: Path, arch: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    best_info = {
        "trial_number": best.number,
        "psnr_dB":      best.value,
        "params":       best.params,
    }
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"\nBest trial #{best.number}  PSNR = {best.value:.4f} dB")
    print("  Params:", json.dumps(best.params, indent=4))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: -(t.value or -999))
    if completed:
        fields = ["rank", "trial", "psnr_dB"] + list(completed[0].params.keys())
        with open(out_dir / "trials_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for rank, t in enumerate(completed, 1):
                row = {"rank": rank, "trial": t.number, "psnr_dB": round(t.value, 4)}
                row.update(t.params)
                w.writerow(row)
        print(f"Saved {len(completed)} completed trials → {out_dir / 'trials_summary.csv'}")

    values = [t.value for t in completed]
    best_so_far, running_best = [], -float("inf")
    for v in values:
        running_best = max(running_best, v)
        best_so_far.append(running_best)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(range(len(values)), values, s=20, alpha=0.5, label="Trial PSNR")
    ax.plot(range(len(best_so_far)), best_so_far, color="tomato", lw=2, label="Best so far")
    ax.set_xlabel("Completed trial index")
    ax.set_ylabel("Val PSNR (dB)")
    ax.set_title(f"Optimisation history — {arch}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "optimization_history.png", dpi=150)
    plt.close()

    if len(completed) >= 4:
        try:
            importances = optuna.importance.get_param_importances(study)
            params_sorted = list(importances.keys())
            vals_sorted   = [importances[p] for p in params_sorted]

            fig, ax = plt.subplots(figsize=(8, max(3, len(params_sorted) * 0.5)))
            ax.barh(params_sorted[::-1], vals_sorted[::-1], color="steelblue")
            ax.set_xlabel("Importance score (FAnova)")
            ax.set_title(f"Hyperparameter importances — {arch}")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(out_dir / "param_importances.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"  [warning] Could not compute param importances: {e}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Optuna HPO — unet_deep (mse) | resnet_16blocks (combined) | "
                    "unet_attention (combined) | diffusion (mse)"
    )
    p.add_argument("--arch", required=True,
                   choices=["unet_deep", "resnet_16blocks", "unet_attention", "diffusion"],
                   help="Architecture to optimise")
    p.add_argument("--n-trials",          type=int,   default=None,
                   help="Number of Optuna trials (default: per-model budget)")
    p.add_argument("--epochs-per-trial",  type=int,   default=None,
                   help="Training epochs per trial (default: per-model budget)")
    p.add_argument("--val-split",         type=float, default=0.05,
                   help="Validation fraction (default: 0.05)")
    p.add_argument("--name",              type=str,   default=None,
                   help="Output subfolder name (default: optuna_<arch>)")
    p.add_argument("--n-startup-trials",  type=int,   default=5,
                   help="Trials before pruning starts (default: 5)")
    p.add_argument("--warmup-epochs",     type=int,   default=15,
                   help="Epochs before pruning starts within a trial (default: 15)")
    p.add_argument("--val-ddim-steps",    type=int,   default=10,
                   help="DDIM steps used during HPO validation (diffusion only, default: 10)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    arch  = args.arch
    name  = args.name or f"optuna_{arch}"
    out_dir = ROOT / "results" / name

    # Resolve compute budget from CLI or per-model defaults
    default_trials, default_epochs = DEFAULT_BUDGET[arch]
    n_trials   = args.n_trials          if args.n_trials          is not None else default_trials
    n_epochs   = args.epochs_per_trial  if args.epochs_per_trial  is not None else default_epochs

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Arch   : {arch}")
    print(f"Trials : {n_trials}  ×  {n_epochs} epochs each")
    print(f"Output : {out_dir}")

    data_cfg = {
        "noisy_path": "data/noisy_train_19k_harder.npy",
        "clean_path": "data/clean_train_19k_harder.npy",
        "val_split":  args.val_split,
    }
    train_ds, val_ds = load_data(data_cfg, ROOT, seed=42)
    print(f"Train  : {len(train_ds)}   Val : {len(val_ds)}")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.warmup_epochs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir / 'study.db'}"

    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    already_done = len([t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
    remaining    = max(0, n_trials - already_done)
    print(f"Already completed: {already_done}  |  Remaining: {remaining}")

    if remaining > 0:
        if arch == "diffusion":
            objective = make_diffusion_objective(
                train_ds, val_ds,
                n_epochs=n_epochs,
                device=device,
                val_ddim_steps=args.val_ddim_steps,
            )
        else:
            objective = make_objective(train_ds, val_ds, arch, n_epochs, device)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    _save_results(study, out_dir, arch)


if __name__ == "__main__":
    main()
