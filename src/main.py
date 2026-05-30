"""
Benchmark runner for lunar image denoising.

This file is mainly the "orchestrator" for the project:
1. read CLI arguments and the YAML config,
2. load the datasets,
3. build each requested model,
4. call `trainer.fit(...)` to actually train it,
5. reload the best checkpoint,
6. evaluate it and save plots / predictions / summaries.

Important: the real training loop is NOT written inline in this file.
The gradient-update logic lives in `src/trainer.py`:
- `fit(...)` owns the epoch loop,
- `_run_epoch(...)` does forward pass, loss, `backward()`, and `optimizer.step()`.

Usage:

    # Run one architecture:
    python src/main.py --run u_net
    python src/main.py --run u_net_attention
    python src/main.py --run res_net

    # Run two architectures:
    python src/main.py --run u_net res_net
    python src/main.py --run u_net u_net_attention 

    # Run everything in the YAML file:
    python src/main.py  
"""

import argparse
from copy import deepcopy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# The project is run as a plain script (`python src/main.py`), so we manually
# add `src/` to the import path and can import sibling modules below.

from dataset import load_data, load_test, make_loaders  # noqa: E402
from models import build_model               # noqa: E402
from trainer import fit                      # noqa: E402
from metrics import compute_metrics          # noqa: E402


def parse_args():
    """Read the small CLI surface for this benchmark runner."""
    p = argparse.ArgumentParser(description="Benchmark denoising architectures")
    # Config controls data paths, model definitions, and training hyperparameters.
    p.add_argument("--config", default=str(ROOT / "src" / "config_next.yaml"),
                   help="Path to config.yaml")
    # Optional filter so we do not have to train every experiment every time.
    p.add_argument("--run", nargs="*", metavar="NAME",
                   help="Only run these experiment names")
    return p.parse_args()


def _merge_nested_dicts(base: dict, override: dict) -> dict:
    """Recursively merge two config dictionaries.

    This lets the YAML define per-family defaults and keep each variant short:
    variant values overwrite only the keys they need to change.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _collect_experiments(cfg: dict) -> list[dict]:
    """Normalize config.yaml into the flat experiment list used by the runner.

    Supported formats:
    - old format: `experiments: [...]`
    - grouped format with one experiment per family
    - grouped format with `defaults` + `variants`
    """
    if "experiments" in cfg:
        return cfg["experiments"]

    groups = cfg.get("experiment_groups")
    if not groups:
        raise ValueError("Config must define either 'experiments' or 'experiment_groups'.")

    experiments = []
    for family_name, family_cfg in groups.items():
        # Simple grouped format:
        # experiment_groups:
        #   u_net:
        #     model: ...
        #     training: ...
        if "variants" not in family_cfg:
            exp = deepcopy(family_cfg)
            exp.setdefault("name", family_name)
            exp.setdefault("family", family_name)
            experiments.append(exp)
            continue

        # Expanded grouped format with shared defaults + multiple variants.
        defaults = family_cfg.get("defaults", {})
        variants = family_cfg.get("variants", [])

        for variant in variants:
            exp = _merge_nested_dicts(defaults, variant)
            if "name" not in exp:
                raise ValueError(f"Experiment in family {family_name!r} is missing a 'name'.")
            exp.setdefault("family", family_name)
            experiments.append(exp)

    return experiments


def _save_comparison(model, train_ds, val_ds, device, out_dir: Path, run_name: str):
    """Save a simple visual sanity check: noisy vs clean vs model output."""

    def _triplet(ds, idx, label, fname):
        # Grab one example directly from the dataset, not from a DataLoader.
        noisy_t, clean_t = ds[idx]

        # Inference mode only: we just want a denoised prediction for plotting.
        model.eval()
        with torch.no_grad():
            pred = model(noisy_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()

        # Side-by-side comparison makes it easy to see if the model is learning
        # to preserve structure while removing noise.
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for ax, img, lbl in zip(
            axes,
            [noisy_t.squeeze().numpy(), clean_t.squeeze().numpy(), pred],
            ["Noisy Input", "Clean Target", "Denoised"],
        ):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(lbl)
            ax.axis("off")
        fig.suptitle(f"{run_name} — {label}")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    # Save one train sample and one validation sample so we can compare both.
    _triplet(train_ds, 0, "Training Sample", "comparison_train.png")
    _triplet(val_ds, 0, "Validation Sample", "comparison_val.png")


def _save_test_predictions(model, test_ds, device, out_dir: Path):
    """Run inference on the blind test set and save predictions as .npy + a sample grid."""
    # This loader is only for inference on the unlabeled/blind test set.
    # There is no loss computation or optimization here.
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            # `test_ds` returns noisy images only, so the model output is the
            # denoised prediction we want to export.
            preds.append(model(batch.to(device)).squeeze(1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)          # (N, H, W) float32 in [0,1]
    np.save(out_dir / "test_predictions.npy", preds)

    # Also save a small visual grid so we can quickly inspect output quality
    # without opening the `.npy` file manually.
    n_show = min(8, len(test_ds))
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2, 4))
    for i in range(n_show):
        axes[0, i].imshow(test_ds[i].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(preds[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Noisy", fontsize=9)
    axes[1, 0].set_ylabel("Denoised", fontsize=9)
    fig.suptitle("Blind test predictions (first 8)")
    plt.tight_layout()
    plt.savefig(out_dir / "test_predictions_grid.png", dpi=150)
    plt.close()
    print(f"  -> Saved test predictions: {out_dir / 'test_predictions.npy'} "
          f"({len(preds)} images)")


def _print_and_save_summary(results: list, output_dir: Path):
    # Write the compact experiment summary to CSV for later comparison/reporting.
    fields = ["name", "architecture", "params", "best_val_loss", "final_val_loss",
              "val_mse", "val_psnr"]
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nSaved: {csv_path}")

    # Build a quick leaderboard chart using validation PSNR.
    names = [r["name"] for r in results]
    psnrs = [r["val_psnr"] for r in results]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    bars = ax.bar(names, psnrs, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Benchmark — Validation PSNR")
    ax.set_ylim(0, max(psnrs) * 1.15)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    chart_path = output_dir / "benchmark_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved: {chart_path}")

    # Print the same comparison in the terminal, sorted by best PSNR first.
    print("\n" + "=" * 76)
    print(f"{'Experiment':<26} {'Arch':<12} {'Params':>9} {'BestVal':>9} {'PSNR dB':>9}")
    print("-" * 76)
    for r in sorted(results, key=lambda x: -x["val_psnr"]):
        print(f"{r['name']:<26} {r['architecture']:<12} {r['params']:>9,} "
              f"{r['best_val_loss']:>9.6f} {r['val_psnr']:>9.2f}")
    print("=" * 76)


def main():
    args = parse_args()

    # Load the global experiment specification from YAML.
    # This file defines:
    # - where the datasets live,
    # - where outputs are written,
    # - which experiments to run,
    # - and each experiment's model/training hyperparameters.
    #
    # The config can be either:
    # - a flat `experiments: [...]` list, or
    # - grouped `experiment_groups` with one experiment per family,
    # - grouped `experiment_groups` with per-family defaults + variants.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seed numpy + torch so train/val splitting and training are reproducible.
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use GPU if available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # All per-experiment folders and summary artifacts will be created here.
    output_dir = ROOT / cfg.get("output_dir", "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the paired noisy/clean dataset and split it into train/validation.
    # `load_test` optionally loads a blind noisy-only set for final inference.
    train_ds, val_ds = load_data(cfg["data"], ROOT, seed)
    test_ds = load_test(cfg["data"], ROOT)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}"
          + (f"  Test (blind): {len(test_ds)}" if test_ds else ""))

    # Convert the YAML into the flat list consumed by the rest of the runner.
    # With grouped configs, each variant inherits defaults from its family.
    experiments = _collect_experiments(cfg)

    # `--run ...` filters by experiment / variant name after normalization.
    if args.run:
        experiments = [e for e in experiments if e["name"] in args.run]
        if not experiments:
            print(f"No experiments matched: {args.run}")
            sys.exit(1)

    # We collect one summary row per experiment and write them all at the end.
    all_results = []

    # Main benchmark loop: each iteration trains and evaluates one architecture /
    # hyperparameter setting from the YAML file.
    for exp in experiments:
        name = exp["name"]
        print(f"\n{'=' * 60}\nExperiment: {name}\n{'=' * 60}")

        # Build fresh DataLoaders for this experiment using the requested batch
        # size. The underlying train/val split stays the same across runs.
        batch_size = int(exp["training"].get("batch_size", 32))
        train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size)

        # Instantiate the model described in `exp["model"]`.
        model = build_model(exp["model"]).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {n_params:,}")

        exp_dir = output_dir / name

        # This is the most important line if you are looking for "where training
        # happens": `fit(...)` launches the real training loop.
        #
        # Inside `src/trainer.py`, `fit(...)` will:
        # - create optimizer / scheduler / loss from `exp["training"]`,
        # - loop over epochs,
        # - call `_run_epoch(..., optimizer)` for training,
        # - call `_run_epoch(..., optimizer=None)` for validation,
        # - save `best_model.pt` whenever validation improves.
        history = fit(model, train_loader, val_loader, exp["training"], device, exp_dir, name)

        # Training above mutates `model` in memory, but for evaluation we
        # explicitly reload the best checkpoint saved during training, not just
        # the final epoch weights. This means metrics are reported on the best
        # validation model seen during the run.
        model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device))
        metrics = compute_metrics(model, val_loader, device)
        print(f"  -> Val MSE: {metrics['mse']:.6f}  PSNR: {metrics['psnr']:.2f} dB")

        # Save qualitative examples from train/val so we can inspect outputs.
        _save_comparison(model, train_ds, val_ds, device, exp_dir, name)

        # If a blind test set exists, export predictions for submission or
        # offline inspection. This is inference only, not training.
        if test_ds is not None:
            _save_test_predictions(model, test_ds, device, exp_dir)

        # Keep a compact in-memory summary for the final benchmark table/CSV.
        result = {
            "name": name,
            "architecture": exp["model"]["architecture"],
            "params": n_params,
            "best_val_loss": history["best_val_loss"],
            "final_val_loss": history["final_val_loss"],
            "val_mse": metrics["mse"],
            "val_psnr": metrics["psnr"],
        }
        all_results.append(result)

        # Save the full per-epoch loss history too, so plots/reports can be
        # rebuilt later without rerunning training.
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(
                {**result,
                 "train_losses": history["train_losses"],
                 "val_losses": history["val_losses"]},
                f, indent=2,
            )

    # After every experiment has finished, write the cross-experiment summary.
    _print_and_save_summary(all_results, output_dir)


if __name__ == "__main__":
    main()
