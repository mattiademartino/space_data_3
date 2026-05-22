"""
Homework 2: Denoising 2D Lunar Images with a U-Net CNN
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BASE_DIR = Path(__file__).resolve().parent

# ─── Dataset ──────────────────────────────────────────────────────────────────
class LunarDataset(Dataset):
    def __init__(self, noisy, clean):
        # Normalise to [0, 1] and add channel dim → (N, 1, 64, 64) float32
        self.noisy = torch.tensor(noisy / 255.0, dtype=torch.float32).unsqueeze(1)
        self.clean = torch.tensor(clean / 255.0, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


# ─── U-Net ────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for single-channel 64×64 image denoising.
    Encoder: 3 downsampling stages (MaxPool).
    Bottleneck: DoubleConv at the lowest resolution.
    Decoder: 3 upsampling stages (bilinear + skip connections).
    """
    def __init__(self, features=(32, 64, 128)):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(1, features[0])
        self.enc2 = DoubleConv(features[0], features[1])
        self.enc3 = DoubleConv(features[1], features[2])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(features[2], features[2] * 2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(features[2] * 2, features[2], 2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0])

        # Output
        self.out_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


# ─── Load data & split ────────────────────────────────────────────────────────
noisy_np = np.load(BASE_DIR / "noisy_images_small_1k (1).npy")
clean_np = np.load(BASE_DIR / "clean_images_small_1k (1).npy")

dataset = LunarDataset(noisy_np, clean_np)

val_size  = int(0.05 * len(dataset))   # 5 % held-out
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(SEED))

BATCH = 32
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {train_size}  |  Val: {val_size}")

# ─── Training ─────────────────────────────────────────────────────────────────
EPOCHS   = 60
LR       = 1e-3
WD       = 1e-4

model     = UNet().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # -- train --
    model.train()
    running = 0.0
    for noisy_b, clean_b in train_loader:
        noisy_b, clean_b = noisy_b.to(DEVICE), clean_b.to(DEVICE)
        pred = model(noisy_b)
        loss = criterion(pred, clean_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * noisy_b.size(0)
    train_losses.append(running / train_size)

    # -- validate --
    model.eval()
    running = 0.0
    with torch.no_grad():
        for noisy_b, clean_b in val_loader:
            noisy_b, clean_b = noisy_b.to(DEVICE), clean_b.to(DEVICE)
            pred = model(noisy_b)
            running += criterion(pred, clean_b).item() * noisy_b.size(0)
    val_losses.append(running / val_size)

    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train MSE: {train_losses[-1]:.6f}  "
              f"val MSE:   {val_losses[-1]:.6f}")

# ─── Plot 1: Training & Validation Loss ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, EPOCHS + 1), train_losses, label="Train MSE")
ax.plot(range(1, EPOCHS + 1), val_losses,   label="Validation MSE", linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training & Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_loss_curves.png", dpi=150)
plt.show()
print("Saved: plot_loss_curves.png")

# ─── Plot 2: Learned Filters (first conv layer) ───────────────────────────────
filters = model.enc1.block[0].weight.detach().cpu().numpy()  # (32, 1, 3, 3)
n_filters = 16   # show first 16
ncols = 8
nrows = n_filters // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
for i, ax in enumerate(axes.flat):
    f = filters[i, 0]   # (3, 3)
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-8)
    ax.imshow(f_norm, cmap="gray", interpolation="nearest")
    ax.axis("off")
fig.suptitle("First 16 Learned Filters — Encoder Conv1 (3×3)", y=1.02)
plt.tight_layout()
plt.savefig("plot_learned_filters.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plot_learned_filters.png")

# ─── Helper: denoise a single sample ──────────────────────────────────────────
def denoise(model, dataset_item):
    noisy_t, clean_t = dataset_item
    model.eval()
    with torch.no_grad():
        pred = model(noisy_t.unsqueeze(0).to(DEVICE)).squeeze().cpu().numpy()
    noisy_img = noisy_t.squeeze().numpy()
    clean_img = clean_t.squeeze().numpy()
    return noisy_img, clean_img, pred


def show_triplet(noisy, clean, pred, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, img, label in zip(axes,
                               [noisy, clean, pred],
                               ["Noisy Input", "Clean Target", "U-Net Denoised"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")


# ─── Plot 3a: Training-set sample ─────────────────────────────────────────────
train_idx = 0   # first sample of the training subset
noisy_img, clean_img, denoised = denoise(model, train_ds[train_idx])
show_triplet(noisy_img, clean_img, denoised,
             "Image Comparison — Training Set Sample",
             "plot_comparison_train.png")

# MSE for this image
mse_train = np.mean((denoised - clean_img) ** 2)
print(f"Per-image MSE (training sample): {mse_train:.6f}")

# ─── Plot 3b: Validation-set sample ───────────────────────────────────────────
val_idx = 0    # first sample of the validation subset
noisy_img_v, clean_img_v, denoised_v = denoise(model, val_ds[val_idx])
show_triplet(noisy_img_v, clean_img_v, denoised_v,
             "Image Comparison — Validation Set Sample",
             "plot_comparison_val.png")

mse_val = np.mean((denoised_v - clean_img_v) ** 2)
print(f"Per-image MSE (validation sample): {mse_val:.6f}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n=== Final Results ===")
print(f"Final train MSE : {train_losses[-1]:.6f}")
print(f"Final val MSE   : {val_losses[-1]:.6f}")
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")
