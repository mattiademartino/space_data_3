---
license: mit
tags:
  - image-denoising
  - unet
  - pytorch
  - lunar-imagery
---

# UNetDeep — Lunar Image Denoiser (unet_deep_01)

4-stage U-Net trained to denoise 64×64 lunar surface images.  
Best model from an Optuna hyperparameter search over 4 architectures × 2 loss functions.

## Results (150 epochs, 5% validation split)

| Metric       | Value       |
|--------------|-------------|
| Val MSE      | 0.001758    |
| Val MAE      | 0.03213     |
| Val PSNR     | **27.55 dB** |

## Architecture

- **Class**: `UNetDeep` — 4-stage encoder-decoder with skip connections
- **Features**: `[64, 128, 256, 512]` (large preset)
- **Bottleneck**: 1024 channels, spatial size 4×4 (for 64×64 input)
- **Dropout**: 0.10887 (Dropout2d after each double-conv block)
- **Output activation**: Sigmoid → output in [0, 1]
- **Parameters**: ~31 M

```
Input (1×64×64)
  enc1: 1  → 64     [64×64]
  enc2: 64 → 128    [32×32]
  enc3: 128→ 256    [16×16]
  enc4: 256→ 512    [8×8]
  bottleneck: 512 → 1024  [4×4]
  dec4: 1024→ 512   [8×8]
  dec3: 512 → 256   [16×16]
  dec2: 256 → 128   [32×32]
  dec1: 128 → 64    [64×64]
Output (1×64×64)
```

## Training hyperparameters

| Parameter      | Value      |
|----------------|------------|
| Loss           | MSE        |
| Optimizer      | AdamW      |
| Learning rate  | 9.664e-4   |
| Weight decay   | 2.472e-3   |
| Batch size     | 16         |
| Epochs         | 150        |
| LR schedule    | Cosine annealing |
| Val split      | 5%         |
| Random seed    | 42         |

Hyperparameters found via Optuna TPE sampler (100 trials per architecture/loss combo).

## Dataset

- **Train**: `noisy_train_19k_harder.npy` / `clean_train_19k_harder.npy` — 19k paired 64×64 grayscale patches
- **Test**: `noisy_val_1k_harder.npy` — 1k noisy patches (blind, no clean labels)
- Pixel values normalized to [0, 1]

## How to use

```python
import torch
import numpy as np
from huggingface_hub import hf_hub_download

# --- load weights ---
pt_path = hf_hub_download(
    repo_id="mattiademartino/unet-deep-lunar-denoiser",
    filename="best_model_mae.pt",
)

# --- build model (copy-paste, no extra dependencies) ---
import torch.nn as nn
from typing import Sequence

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0: layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class UNetDeep(nn.Module):
    def __init__(self, features=(64,128,256,512), dropout=0.10887):
        super().__init__()
        f = list(features)
        self.pool = nn.MaxPool2d(2)
        self.enc1, self.enc2 = DoubleConv(1,f[0],dropout), DoubleConv(f[0],f[1],dropout)
        self.enc3, self.enc4 = DoubleConv(f[1],f[2],dropout), DoubleConv(f[2],f[3],dropout)
        self.bottleneck = DoubleConv(f[3],f[3]*2,dropout)
        self.up4, self.dec4 = nn.ConvTranspose2d(f[3]*2,f[3],2,stride=2), DoubleConv(f[3]*2,f[3],dropout)
        self.up3, self.dec3 = nn.ConvTranspose2d(f[3],f[2],2,stride=2),   DoubleConv(f[2]*2,f[2],dropout)
        self.up2, self.dec2 = nn.ConvTranspose2d(f[2],f[1],2,stride=2),   DoubleConv(f[1]*2,f[1],dropout)
        self.up1, self.dec1 = nn.ConvTranspose2d(f[1],f[0],2,stride=2),   DoubleConv(f[0]*2,f[0],dropout)
        self.out_conv = nn.Conv2d(f[0],1,1)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(b),e4],1)); d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1)); d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return torch.sigmoid(self.out_conv(d1))

model = UNetDeep()
model.load_state_dict(torch.load(pt_path, map_location="cpu"))
model.eval()

# --- inference on a single patch ---
noisy_patch = np.load("noisy_val_1k_harder.npy")[0].astype(np.float32) / 255.0
x = torch.from_numpy(noisy_patch).unsqueeze(0).unsqueeze(0)  # (1,1,64,64)
with torch.no_grad():
    denoised = model(x).squeeze().numpy()  # (64,64) in [0,1]
```

Or use the full `submission.py` script (see repo):

```bash
python submission.py test --data-dir data/ --out predictions.npy
# automatically downloads this model from Hugging Face
```

## Reproduce training

```bash
git clone <repo>
pip install torch numpy matplotlib
python submission.py train --data-dir data/ --out-dir results/
```

Requires `noisy_train_19k_harder.npy` and `clean_train_19k_harder.npy` in `data/`.
