#!/bin/bash
# ============================================================
# SLURM array job — Final training for models 1-8
# (unet_deep x3, unet_attention x3, resnet x2)
#
# Submit:  sbatch run_final.sh
# ============================================================
#SBATCH --job-name=final_train
#SBATCH --array=1-8
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=slurm_logs/final_%A_%a.log

source startup.sh

echo "========================================"
echo "Final training — model $SLURM_ARRAY_TASK_ID / 10"
echo "========================================"

python src/train_final.py --model-id "$SLURM_ARRAY_TASK_ID"
