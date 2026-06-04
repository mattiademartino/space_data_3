#!/bin/bash
# ============================================================
# SLURM array job — Final training for models 9-10 (diffusion)
# Diffusion training is slower (T=1000, DDIM validation) → 12h
#
# Submit:  sbatch run_final_diffusion.sh
# ============================================================
#SBATCH --job-name=final_diffusion
#SBATCH --array=9-10
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=slurm_logs/final_%A_%a.log

source startup.sh

echo "========================================"
echo "Final training (diffusion) — model $SLURM_ARRAY_TASK_ID / 10"
echo "========================================"

python src/train_final.py --model-id "$SLURM_ARRAY_TASK_ID"
