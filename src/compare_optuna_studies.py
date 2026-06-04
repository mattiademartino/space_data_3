"""
Compare best trials across all Optuna studies by PSNR and validation loss.

For studies that already have best_val_loss stored as user_attr, it reads it directly.
For older studies (MSE-only), val MSE is derivable from PSNR: MSE = 10^(-PSNR/10).
For combined-loss studies without the attr, val_loss is unavailable without retraining.

Usage:
    python src/compare_optuna_studies.py
    python src/compare_optuna_studies.py --top 5      # show top-N trials per arch
    python src/compare_optuna_studies.py --no-plot
"""

import argparse
import math
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent

ARCHS = ["unet_deep", "resnet_16blocks", "unet_attention", "diffusion"]
LOSS_TYPES = {
    "unet_deep":       "mse",
    "resnet_16blocks": "combined",
    "unet_attention":  "combined",
    "diffusion":       "mse",
}


def load_study(arch: str):
    db = ROOT / "results" / f"optuna_{arch}" / "study.db"
    if not db.exists():
        return None
    storage = f"sqlite:///{db}"
    return optuna.load_study(study_name=f"optuna_{arch}", storage=storage)


def best_val_loss_for_trial(trial, loss_type: str):
    """Return (val_loss, source) where source is 'stored' or 'derived' or 'unavailable'."""
    stored = trial.user_attrs.get("best_val_loss", None)
    if stored is not None:
        return stored, "stored"
    if loss_type == "mse" and trial.value is not None:
        # PSNR = 10*log10(1/MSE)  =>  MSE = 10^(-PSNR/10)
        return 10 ** (-trial.value / 10.0), "derived from PSNR"
    return None, "unavailable"


def print_summary(top_n: int = 3):
    print(f"\n{'='*72}")
    print(f"{'Arch':<20} {'#Trials':>7} {'Best PSNR (dB)':>14} {'Val Loss':>12}  {'Source'}")
    print(f"{'='*72}")

    rows = []
    for arch in ARCHS:
        study = load_study(arch)
        if study is None:
            print(f"{arch:<20}  (no study.db found)")
            continue
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print(f"{arch:<20}  (no completed trials)")
            continue

        best = study.best_trial
        loss_type = LOSS_TYPES[arch]
        val_loss, source = best_val_loss_for_trial(best, loss_type)

        val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
        print(
            f"{arch:<20} {len(completed):>7} {best.value:>14.4f} {val_loss_str:>12}  [{source}]"
        )
        rows.append((arch, len(completed), best.value, val_loss, source, best, loss_type))

    print(f"{'='*72}")
    print("\nNote: 'derived from PSNR' = MSE = 10^(-PSNR/10), valid only for MSE-trained models.")
    print("      'unavailable' = combined-loss model from an old run — rerun to populate.\n")

    # Per-arch top-N table
    for arch in ARCHS:
        study = load_study(arch)
        if study is None:
            continue
        loss_type = LOSS_TYPES[arch]
        completed = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: -(t.value or -999),
        )
        if not completed:
            continue
        print(f"  Top-{top_n} trials — {arch}  (loss: {loss_type})")
        print(f"  {'Rank':>4} {'Trial':>5} {'PSNR (dB)':>10} {'Val Loss':>12}  {'Source'}")
        print(f"  {'-'*50}")
        for rank, t in enumerate(completed[:top_n], 1):
            vl, src = best_val_loss_for_trial(t, loss_type)
            vl_str = f"{vl:.6f}" if vl is not None else "N/A"
            print(f"  {rank:>4} {t.number:>5} {t.value:>10.4f} {vl_str:>12}  [{src}]")
        print()

    return rows


def plot_comparison(rows, out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    archs     = [r[0] for r in rows]
    psnrs     = [r[2] for r in rows]
    val_losses = [r[3] for r in rows]

    x = np.arange(len(archs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR bar chart
    bars = axes[0].bar(x, psnrs, color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(archs, rotation=15, ha="right")
    axes[0].set_ylabel("Best Val PSNR (dB)")
    axes[0].set_title("PSNR Comparison (best trial per arch)")
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Val Loss bar chart (only where available)
    available = [(i, r) for i, r in enumerate(rows) if r[3] is not None]
    if available:
        xi, ri = zip(*available)
        vl = [r[3] for r in ri]
        arches_avail = [r[0] for r in ri]
        colors = ["tomato" if r[4] == "stored" else "orange" for r in ri]
        bars2 = axes[1].bar(range(len(xi)), vl, color=colors, alpha=0.8)
        axes[1].set_xticks(range(len(xi)))
        axes[1].set_xticklabels(arches_avail, rotation=15, ha="right")
        axes[1].set_ylabel("Best Val Loss")
        axes[1].set_title("Val Loss Comparison (best trial per arch)")
        axes[1].bar_label(bars2, fmt="%.4f", padding=3)
        axes[1].grid(True, axis="y", alpha=0.3)

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="tomato",  alpha=0.8, label="stored (tracked)"),
            Patch(facecolor="orange",  alpha=0.8, label="derived from PSNR"),
        ]
        axes[1].legend(handles=legend_els, loc="upper right", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No val loss data available.\nRerun studies to populate.",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top",     type=int,  default=3,     help="Top-N trials to show per arch")
    p.add_argument("--no-plot", action="store_true",       help="Skip plot generation")
    args = p.parse_args()

    rows = print_summary(top_n=args.top)

    if not args.no_plot and rows:
        out = ROOT / "results" / "optuna_comparison.png"
        plot_comparison(rows, out)


if __name__ == "__main__":
    main()
