"""
Visualize .npy denoising results.

Usage:
    python visualize_results.py --noisy data/noisy.npy --preds results/test_predictions.npy
    python visualize_results.py --noisy data/noisy.npy --clean data/clean.npy --preds results/test_predictions.npy
    python visualize_results.py --noisy data/noisy.npy --clean data/clean.npy --preds results/test_predictions.npy --n 8 --idx 0 5 10 15
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy",  required=True,  help="Noisy input .npy")
    p.add_argument("--preds",  required=True,  help="Model predictions .npy")
    p.add_argument("--clean",  default=None,   help="Clean ground truth .npy (optional)")
    p.add_argument("--n",      type=int, default=8, help="Number of samples to show")
    p.add_argument("--idx",    type=int, nargs="*", default=None,
                   help="Specific indices to show (overrides --n)")
    p.add_argument("--out",    default="visualization.png", help="Output file")
    return p.parse_args()


def load(path):
    arr = np.load(path).astype(np.float32)
    if arr.max() > 2.0:
        arr /= 255.0
    # Strip channel dim if present: (N,1,H,W) → (N,H,W)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr.squeeze(1)
    return arr


def main():
    args  = parse_args()
    noisy = load(args.noisy)
    preds = load(args.preds)
    clean = load(args.clean) if args.clean else None

    indices = args.idx if args.idx else list(range(min(args.n, len(noisy))))
    n       = len(indices)
    n_cols  = n
    n_rows  = 3 if clean is not None else 2
    labels  = ["Noisy", "Denoised", "Clean"] if clean is not None else ["Noisy", "Denoised"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, idx in enumerate(indices):
        images = [noisy[idx], preds[idx]]
        if clean is not None:
            images.append(clean[idx])
        for row, img in enumerate(images):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=9)
        axes[0, col].set_title(f"#{idx}", fontsize=8)

    plt.suptitle("Denoising Results", y=1.01)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()