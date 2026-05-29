import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from main import ROOT, _collect_experiments
from metrics import mse, psnr
from models import build_model
from trainer import build_criterion


class BenchmarkDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray | None = None):
        noisy = noisy.astype(np.float32)
        if noisy.max() > 2.0:
            noisy /= 255.0
        self.noisy = torch.from_numpy(noisy).unsqueeze(1)

        self.clean = None
        if clean is not None:
            clean = clean.astype(np.float32)
            if clean.max() > 2.0:
                clean /= 255.0
            self.clean = torch.from_numpy(clean).unsqueeze(1)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        if self.clean is None:
            return self.noisy[idx]
        return self.noisy[idx], self.clean[idx]


def parse_args():
    p = argparse.ArgumentParser(description="Run saved checkpoints on a benchmark dataset.")
    p.add_argument(
        "--config",
        nargs="+",
        default=[str(ROOT / "src" / "config_initial.yaml")],
        help="One or more config files used to define experiment architectures.",
    )
    p.add_argument(
        "--noisy-path",
        default=str(ROOT / "data" / "noisy_val_1k_harder.npy"),
        help="Path to the noisy benchmark dataset.",
    )
    p.add_argument(
        "--clean-path",
        default=None,
        help="Optional clean targets for the benchmark. If omitted, no loss/metrics are computed.",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "benchmark_reval"),
        help="Directory where fresh benchmark predictions and summaries are stored.",
    )
    p.add_argument(
        "--run",
        nargs="*",
        metavar="NAME",
        help="Only re-benchmark these experiment names.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for the benchmark pass.",
    )
    return p.parse_args()


def _load_experiments(config_paths: list[str]) -> list[dict]:
    experiments = []
    for cfg_path in config_paths:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        output_dir = cfg.get("output_dir", "results")
        for exp in _collect_experiments(cfg):
            experiments.append(
                {
                    **exp,
                    "_config_path": cfg_path,
                    "_results_dir": str(ROOT / output_dir / exp["name"]),
                }
            )

    # `unet_nvidia_loss` exists in the saved results but is currently commented
    # out in the main config, so we register it here explicitly.
    existing_names = {exp["name"] for exp in experiments}
    if "unet_nvidia_loss" not in existing_names:
        experiments.append(
            {
                "name": "unet_nvidia_loss",
                "model": {
                    "architecture": "unet",
                    "features": [32, 64, 128],
                    "dropout": 0.0,
                },
                "training": {
                    "loss": "nvidia_loss",
                },
                "_config_path": "<manual>",
                "_results_dir": str(ROOT / "results" / "unet_nvidia_loss"),
            }
        )

    return experiments


def _run_inference(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            preds.append(model(batch.to(device)).squeeze(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _compute_benchmark_loss(model, loader, criterion, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for noisy_b, clean_b in loader:
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)
            pred = model(noisy_b)
            loss = criterion(pred, clean_b)
            total_loss += loss.item() * noisy_b.size(0)
            total_count += noisy_b.size(0)
            all_preds.append(pred.cpu())
            all_targets.append(clean_b.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return total_loss / total_count, mse(preds, targets), psnr(preds, targets)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    noisy = np.load(args.noisy_path)
    clean = np.load(args.clean_path) if args.clean_path else None
    if clean is not None and len(noisy) != len(clean):
        raise ValueError("Noisy and clean benchmark arrays must have the same length.")

    experiments = _load_experiments(args.config)
    if args.run:
        requested = set(args.run)
        experiments = [exp for exp in experiments if exp["name"] in requested]

    if not experiments:
        raise ValueError("No experiments selected.")

    ds = BenchmarkDataset(noisy, clean)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for exp in experiments:
        exp_dir = Path(exp["_results_dir"])
        checkpoint = exp_dir / "best_model.pt"
        if not checkpoint.exists():
            print(f"Skipping {exp['name']}: checkpoint not found at {checkpoint}")
            continue

        print(f"\n{'=' * 60}\nBenchmarking: {exp['name']}\n{'=' * 60}")
        model = build_model(exp["model"]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))

        preds = _run_inference(model, loader, device)
        pred_dir = output_dir / exp["name"]
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / "benchmark_predictions.npy"
        np.save(pred_path, preds)

        row = {
            "name": exp["name"],
            "architecture": exp["model"]["architecture"],
            "checkpoint": str(checkpoint),
            "config": exp["_config_path"],
            "prediction_path": str(pred_path),
            "benchmark_loss": "",
            "benchmark_mse": "",
            "benchmark_psnr": "",
        }

        if clean is not None:
            criterion = build_criterion(exp["training"])
            bench_loss, bench_mse, bench_psnr = _compute_benchmark_loss(model, loader, criterion, device)
            row["benchmark_loss"] = bench_loss
            row["benchmark_mse"] = bench_mse
            row["benchmark_psnr"] = bench_psnr
            print(
                f"  -> benchmark loss: {bench_loss:.6f}  "
                f"MSE: {bench_mse:.6f}  PSNR: {bench_psnr:.2f} dB"
            )
        else:
            print(f"  -> saved predictions: {pred_path}")

        with open(pred_dir / "benchmark_summary.json", "w") as f:
            json.dump(row, f, indent=2)
        summary_rows.append(row)

    csv_path = output_dir / "benchmark_summary.csv"
    fields = [
        "name",
        "architecture",
        "checkpoint",
        "config",
        "prediction_path",
        "benchmark_loss",
        "benchmark_mse",
        "benchmark_psnr",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {csv_path}")


if __name__ == "__main__":
    main()
