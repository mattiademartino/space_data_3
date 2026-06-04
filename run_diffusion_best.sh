#!/bin/bash
# ============================================================
# Full training run for diffusion — best params from Optuna HPO
#
# Best trial #0: PSNR = 25.53 dB (30 epochs, MSE loss, 10 DDIM steps)
# features=large [64,128,256]  T=1000  t_dim=128
# lr=4.25e-4  optimizer=adamw  batch_size=32
#
# This run uses more epochs (150) and mix loss (MS-SSIM+L1)
# for better perceptual quality, with 50 DDIM steps at inference.
# ============================================================

#SBATCH --job-name=diffusion_best
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/diffusion_best_%j.log

source startup.sh

python src/diffusion_main.py \
    --epochs     150           \
    --features   64 128 256    \
    --T          1000          \
    --t-dim      128           \
    --lr         0.000425      \
    --batch-size 32            \
    --loss       mix           \
    --ddim-steps 50            \
    --name       diffusion_best
