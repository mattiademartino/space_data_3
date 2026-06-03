#!/bin/bash
# ============================================================
# SLURM job script — Optuna hyperparameter optimisation
#
# Submitting (one job per architecture):
#   sbatch run_optuna.sh unet_deep
#   sbatch run_optuna.sh resnet_16blocks
#   sbatch run_optuna.sh unet_attention
#   sbatch run_optuna.sh diffusion
#
# The architecture is passed as the first argument ($1).
# Default: unet_deep
#
# Budget defaults (overridable via extra args):
#   unet_deep / resnet_16blocks / unet_attention : 40 trials × 40 epochs
#   diffusion                                    : 15 trials × 30 epochs
#
# Interrupted studies resume automatically (SQLite storage).
# ============================================================

#SBATCH --job-name=optuna_hpo
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/optuna_%A_%x.log

source startup.sh

ARCH=${1:-unet_deep}

echo "========================================"
echo "Optuna HPO — arch: $ARCH"
echo "========================================"

python src/optuna_search.py \
    --arch            "$ARCH"      \
    --val-split       0.05         \
    --n-startup-trials 5           \
    --warmup-epochs   15           \
    --name            "optuna_${ARCH}"
