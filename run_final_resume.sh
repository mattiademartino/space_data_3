#!/bin/bash
# Rilancia solo i modelli 1-4,7,8 che sono stati killati per time limit.
# Riprende automaticamente dal checkpoint salvato.
#SBATCH --job-name=final_resume
#SBATCH --array=1-4,7,8
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=slurm_logs/final_resume_%A_%a.log

source startup.sh

echo "========================================"
echo "Final training RESUME — model $SLURM_ARRAY_TASK_ID"
echo "========================================"

python src/train_final.py --model-id "$SLURM_ARRAY_TASK_ID"
