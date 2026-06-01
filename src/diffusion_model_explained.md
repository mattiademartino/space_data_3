# Conditional DDPM for Image Denoising — How It Works

## Overview

This document explains the Conditional Denoising Diffusion Probabilistic Model (DDPM) implemented
in `diffusion_model.py`, `diffusion_trainer.py`, and `diffusion_main.py`.

The model is a **generative approach** to image denoising: instead of directly mapping a noisy
image to a clean image in a single deterministic step (as a standard U-Net does), the diffusion
model learns to iteratively refine a random noise sample into a clean image, **conditioned on the
sensor-noisy observation**.

---

## 1. Why Diffusion for Denoising?

A standard supervised regression model (U-Net, ResNet) trained with MSE minimises the expected
squared error over all possible clean images compatible with the noisy input. When several clean
images are plausible for the same noisy observation, the model tends to output their average —
producing a blurry, over-smoothed result.

The loss function paper (Zhao et al., 2017) already identified this problem: ℓ₂ produces "splotchy
artifacts in flat regions" because the model averages over multiple valid solutions. Switching to
ℓ₁ or MS-SSIM + ℓ₁ (Mix loss) improves perceptual quality, but these are still single-step
deterministic estimators.

A **diffusion model solves this by design**: it is a generative model that samples from the
full conditional distribution p(x_clean | x_noisy), rather than computing a point estimate.
This naturally avoids averaging artefacts.

---

## 2. The Forward Diffusion Process

The forward process defines a sequence of increasingly noisy versions of a clean image x₀.
Given a schedule of noise variances β₁ < β₂ < ... < βᴛ (linearly spaced between 10⁻⁴ and 0.02),
the image at diffusion step t is:

```
x_t = sqrt(ᾱ_t) · x₀  +  sqrt(1 − ᾱ_t) · ε,     ε ~ N(0, I)
```

where ᾱ_t = ∏ᵢ₌₁ᵗ (1 − βᵢ) is the cumulative noise coefficient.

- At t = 0, ᾱ₀ ≈ 1, so x₀ ≈ x₀ (no noise).
- At t = T (= 1000), ᾱᴛ ≈ 0, so xᴛ ≈ ε (pure Gaussian noise).

**This process is fixed** — there are no learnable parameters in the forward direction.

---

## 3. The Model: Conditional DDPM U-Net (`CondDDPMUNet`)

The learnable component is a U-Net that, given:
- **x_t** : the corrupted image at diffusion step t
- **y**   : the original sensor-noisy lunar image (conditioning signal)
- **t**   : the current timestep (encoded as a sinusoidal embedding)

predicts the **original clean image x₀** directly (x₀-parameterisation).

### Architecture

```
Input: [x_t || y]  →  2 channels (concatenated along channel axis)

Encoder:
  enc1: 2 → 32     DoubleConv (Conv-BN-ReLU × 2)
  enc2: 32 → 64    DoubleConv   (after MaxPool)
  enc3: 64 → 128   DoubleConv   (after MaxPool)

Bottleneck:
  128 → 256         DoubleConv   (after MaxPool)
  + time_emb(t)     Sinusoidal embedding → 2-layer MLP → 256 dims
                    Added to bottleneck feature map

Decoder (with attention gates on all skip connections):
  up3 + att3 + dec3:  256 → 128
  up2 + att2 + dec2:  128 → 64
  up1 + att1 + dec1:  64  → 32

Output:
  Conv 32 → 1  +  Sigmoid  →  x₀_pred ∈ [0, 1]
```

The **attention gates** (Oktay et al., 2018) let the decoder focus on the most
informative spatial regions of each encoder skip connection, guided by the bottleneck
signal. This is especially useful for denoising because the model can learn to weight
skip connections based on local image structure.

The **sinusoidal timestep embedding** tells the model how noisy x_t is. At high t the
signal is almost pure noise and the prediction must be approximate; at low t it is
nearly clean and the prediction must be precise. Without this signal, the model cannot
distinguish the two regimes.

---

## 4. Training

At each training step:

1. Sample a **random timestep** t ~ Uniform[0, T) for each image in the batch.
2. Sample Gaussian noise ε ~ N(0, I).
3. Compute the corrupted image: `x_t = sqrt(ᾱ_t) · x₀ + sqrt(1 − ᾱ_t) · ε`.
4. Run the model: `x₀_pred = model(x_t, y, t)`.
5. Compute the **loss in pixel space** between x₀_pred and the true x₀.

The loss can be:
- **L1** (default): fast, robust to outliers, preferred for high-noise timesteps.
- **MSE**: baseline, tends to over-smooth.
- **Mix** (MS-SSIM + L1, α = 0.84): best perceptual quality as shown by Zhao et al. 2017,
  but more expensive (MS-SSIM is evaluated at 5 scales).

Unlike standard DDPM which trains the model to predict the noise ε (ε-parameterisation),
we use **x₀-parameterisation**: the model directly predicts the clean image. This allows
perceptual losses (SSIM, L1) to be applied directly in pixel space, connecting the diffusion
training objective to the findings of the loss function paper.

---

## 5. Inference — DDIM Sampling

Naive DDPM inference requires T = 1000 sequential model evaluations. **DDIM** (Song et al., 2021)
re-interprets the reverse process as a non-Markovian deterministic ODE, reducing the required
steps to n_steps ≈ 50 with minimal quality loss.

### DDIM Algorithm (η = 0, deterministic)

```
x_T ~ N(0, I)

for t = T-1, t_{n-1}, ..., 0   (n_steps evenly-spaced steps):

    # 1. Predict clean image
    x₀_hat = model(x_t, y, t).clamp(0, 1)

    # 2. Reconstruct the predicted noise direction
    ε_hat  = (x_t − sqrt(ᾱ_t) · x₀_hat) / sqrt(1 − ᾱ_t)

    # 3. Jump to the previous timestep (deterministic, no added noise)
    x_{t_prev} = sqrt(ᾱ_{t_prev}) · x₀_hat  +  sqrt(1 − ᾱ_{t_prev}) · ε_hat

return x_0
```

The key property is that with η = 0 (no stochastic noise added at each step), the same
conditioning image y always produces the same denoised output — inference is **deterministic
and reproducible**, which is important for a scientific application.

---

## 6. Connection to the Loss Function Paper

This implementation directly extends the findings of Zhao et al. 2017:

| Model                     | Training objective     | Degeneracy |
|---------------------------|------------------------|------------|
| U-Net + MSE               | ℓ₂ on x₀              | Blurry mean |
| U-Net + L1                | ℓ₁ on x₀              | Less blurry, still deterministic |
| U-Net + Mix               | MS-SSIM + ℓ₁ on x₀    | Best single-step estimate |
| **Conditional DDPM + L1** | ℓ₁ on x₀ **across all t** | No mean averaging by design |
| **Conditional DDPM + Mix**| MS-SSIM+ℓ₁ on x₀ at all t | Perceptually optimal |

The diffusion model is the natural next step: if ℓ₁ > ℓ₂ and Mix > ℓ₁ because they move
away from the deterministic "averaged" solution, the diffusion model goes one step further
by **not committing to a single output** — it samples from the posterior distribution.

---

## 7. Model Size — Is It Large Enough?

The table below compares parameter counts across all architectures in this project.
The DDPM backbone (`CondDDPMUNet`) is parameterised the same way as the existing U-Nets,
so the comparison is direct and fair.

| Model                         | Parameters  | Notes |
|-------------------------------|-------------|-------|
| `unet_baseline` [32,64,128]   | 1,926,433   | homework-2 baseline |
| `unet_attention` [32,64,128]  | 1,948,730   | best single-step model |
| `res_unet` [32,64,128]        | 2,012,481   |       |
| `resnet_16blocks` (64ch)      | 1,185,025   | flat, no downsampling |
| `unet_large` [64,128,256]     | 7,699,009   |       |
| `unet_attention` [64,128,256] | 7,786,602   | largest U-Net tried  |
| **`ddpm_small` [32,64,128]**  | **2,047,834** | ← **default, recommended** |
| **`ddpm_medium` [64,128,256]**| **8,115,882** | ← larger ablation |
| `ddpm_large` [128,256,512]    | 32,312,650  | likely overkill for 128×128 |

**Verdict: the default `[32,64,128]` configuration is well-matched to the task.**

- It sits in the same capacity bracket (~2 M params) as `unet_attention_med`, which is
  your best-performing deterministic model.
- Unlike a standard U-Net, the DDPM backbone handles a 2-channel input and a timestep
  embedding, so the extra ~100 k params over `unet_attention_med` are fully justified.
- The `[64,128,256]` variant (~8 M) is a reasonable ablation if you want to check whether
  more capacity helps; it matches `unet_attention_large` exactly.
- The `[128,256,512]` variant is almost certainly overkill for 128×128 grayscale patches.

**Recommended workflow:**
1. Run `[32,64,128]` with `--loss l1` for 100 epochs as the baseline DDPM experiment.
2. If time permits, run `[64,128,256]` with `--loss mix` for 200 epochs as the "best effort" run.

---

## 8. File Structure

| File                      | Responsibility                                      |
|---------------------------|-----------------------------------------------------|
| `diffusion_model.py`      | `CondDDPMUNet` (backbone) + `GaussianDiffusion` (process wrapper) |
| `diffusion_trainer.py`    | Training loop, validation, loss factory            |
| `diffusion_main.py`       | Data loading, CLI, orchestration, result saving    |
| `diffusion_model_explained.md` | This document                                 |

---

## 9. How to Run on the Euler Cluster (SLURM)

### Step 1 — Create the SLURM batch script

Create a file `run_diffusion.sh` in the project root:

```bash
#!/bin/bash
#SBATCH --job-name=ddpm_denoising
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/diffusion_%j.log

source startup.sh

# --- Experiment 1: default size, L1 loss (fast, ~1h on A100) ---
python src/diffusion_main.py \
    --epochs 100 \
    --loss l1 \
    --features 32 64 128 \
    --ddim-steps 50 \
    --name conditional_ddpm_l1

# --- Experiment 2: default size, Mix loss (best perceptual quality, ~2h on A100) ---
# python src/diffusion_main.py \
#     --epochs 200 \
#     --loss mix \
#     --features 32 64 128 \
#     --ddim-steps 50 \
#     --name conditional_ddpm_mix

# --- Experiment 3: larger backbone, Mix loss (~4h on A100) ---
# python src/diffusion_main.py \
#     --epochs 200 \
#     --loss mix \
#     --features 64 128 256 \
#     --ddim-steps 100 \
#     --name conditional_ddpm_large_mix
```

### Step 2 — Submit the job

```bash
# From the project root on Euler:
sbatch run_diffusion.sh

# Monitor the job:
squeue --me

# Watch the log live:
tail -f slurm_logs/diffusion_<jobid>.log
```

### Step 3 — Check results

After the job finishes, results are in `results/<name>/`:

```
results/conditional_ddpm_l1/
├── best_model.pt              ← best checkpoint (highest val PSNR)
├── final_model.pt             ← weights after last epoch
├── loss_curves.png            ← training loss + val PSNR over epochs
├── comparison_train.png       ← noisy / clean / denoised (train sample)
├── comparison_val.png         ← noisy / clean / denoised (val sample)
├── test_predictions.npy       ← blind test set predictions (N, H, W)
├── test_predictions_grid.png  ← visual grid of 8 test predictions
└── metrics.json               ← full summary (PSNR, MSE, hyperparams)
```

### Estimated wall-clock times on Euler GPUs

| Configuration                      | A100 (est.) | V100 (est.) |
|------------------------------------|-------------|-------------|
| `[32,64,128]`, L1, 100 epochs      | ~50 min     | ~90 min     |
| `[32,64,128]`, Mix, 200 epochs     | ~2 h        | ~3.5 h      |
| `[64,128,256]`, Mix, 200 epochs    | ~5 h        | ~8 h        |

> **Note:** DDIM validation (run every 5 epochs with 20 steps) is the main overhead
> beyond training. If wall-clock time is tight, increase `val_every` in
> `diffusion_trainer.py` or reduce `val_ddim_steps`.

---

## 10. Key Hyperparameters

| CLI flag        | Default     | Description                                           |
|-----------------|-------------|-------------------------------------------------------|
| `--T`           | 1000        | Diffusion timesteps (do not change unless experimenting) |
| `--ddim-steps`  | 50          | DDIM inference steps — 50 is the speed/quality sweet spot |
| `--loss`        | `l1`        | Training loss: `l1` (fast) · `mse` (ablation) · `mix` (best quality) |
| `--features`    | `32 64 128` | U-Net channel widths — controls capacity              |
| `--t-dim`       | 128         | Sinusoidal timestep embedding dimension               |
| `--epochs`      | 50          | **Recommended: 100+ for competitive PSNR**            |
| `--lr`          | `5e-4`      | Learning rate (AdamW + cosine annealing)              |
| `--batch-size`  | 32          | Increase to 64 if GPU memory allows                  |
| `--name`        | `conditional_ddpm` | Output folder under `results/`              |
