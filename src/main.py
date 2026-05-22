"""
Benchmark runner for lunar image denoising.

Usage:
    python src/main.py                              # uses src/config.yaml
    python src/main.py --config src/config.yaml
    python src/main.py --run unet_baseline res_unet  # selective runs
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import load_data, make_loaders  # noqa: E402
from models import build_model               # noqa: E402
from trainer import fit                      # noqa: E402
from metrics import compute_metrics          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark denoising architectures")
    p.add_argument("--config", default=str(ROOT / "src" / "config.yaml"),
                   help="Path to config.yaml")
    p.add_argument("--run", nargs="*", metavar="NAME",
                   help="Only run these experiment names")
    return p.parse_args()


def _save_comparison(model, train_ds, val_ds, device, out_dir: Path, run_name: str):
    def _triplet(ds, idx, label, fname):
        noisy_t, clean_t = ds[idx]
        model.eval()
        with torch.no_grad():
            pred = model(noisy_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()
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

    _triplet(train_ds, 0, "Training Sample", "comparison_train.png")
    _triplet(val_ds, 0, "Validation Sample", "comparison_val.png")


def _print_and_save_summary(results: list, output_dir: Path):
    # CSV
    fields = ["name", "architecture", "params", "best_val_loss", "final_val_loss",
              "val_mse", "val_psnr"]
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nSaved: {csv_path}")

    # Bar chart — PSNR
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

    # Console table
    print("\n" + "=" * 76)
    print(f"{'Experiment':<26} {'Arch':<12} {'Params':>9} {'BestVal':>9} {'PSNR dB':>9}")
    print("-" * 76)
    for r in sorted(results, key=lambda x: -x["val_psnr"]):
        print(f"{r['name']:<26} {r['architecture']:<12} {r['params']:>9,} "
              f"{r['best_val_loss']:>9.6f} {r['val_psnr']:>9.2f}")
    print("=" * 76)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = ROOT / cfg.get("output_dir", "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_data(cfg["data"], ROOT, seed)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    experiments = cfg["experiments"]
    if args.run:
        experiments = [e for e in experiments if e["name"] in args.run]
        if not experiments:
            print(f"No experiments matched: {args.run}")
            sys.exit(1)

    all_results = []

    for exp in experiments:
        name = exp["name"]
        print(f"\n{'=' * 60}\nExperiment: {name}\n{'=' * 60}")

        batch_size = int(exp["training"].get("batch_size", 32))
        train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size)

        model = build_model(exp["model"]).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {n_params:,}")

        exp_dir = output_dir / name
        history = fit(model, train_loader, val_loader, exp["training"], device, exp_dir, name)

        # Evaluate best checkpoint
        model.load_state_dict(torch.load(exp_dir / "best_model.pt", map_location=device))
        metrics = compute_metrics(model, val_loader, device)
        print(f"  -> Val MSE: {metrics['mse']:.6f}  PSNR: {metrics['psnr']:.2f} dB")

        _save_comparison(model, train_ds, val_ds, device, exp_dir, name)

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

        # Per-experiment full metrics JSON
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(
                {**result,
                 "train_losses": history["train_losses"],
                 "val_losses": history["val_losses"]},
                f, indent=2,
            )

    _print_and_save_summary(all_results, output_dir)


if __name__ == "__main__":
    main()
