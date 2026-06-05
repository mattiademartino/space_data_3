#!/bin/bash
# ---------------------------------------------------------------------------
# submit_kfold.sh
#
# Submits one SLURM job per fold for k-fold cross-validation of unet_deep_01.
#
# Usage:
#   bash submit_kfold.sh              # 5 folds, 150 epochs
#   bash submit_kfold.sh 7            # 7 folds, 150 epochs
#   bash submit_kfold.sh 5 200        # 5 folds, 200 epochs
# ---------------------------------------------------------------------------

N_FOLDS=${1:-5}
EPOCHS=${2:-150}

echo "Submitting ${N_FOLDS} k-fold jobs  (${EPOCHS} epochs each)"
echo "-------------------------------------------------------"

for (( FOLD=0; FOLD<N_FOLDS; FOLD++ )); do

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=k_fold_run_${FOLD}_of_${N_FOLDS}
#SBATCH -n 10
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/k_fold_run_${FOLD}_of_${N_FOLDS}_%j.log

source /cluster/home/mriestere/space_data_3/venv/bin/activate

python src/train_kfold_single.py \
    --fold ${FOLD} \
    --n-folds ${N_FOLDS} \
    --epochs ${EPOCHS}
EOF

    echo "  Submitted fold ${FOLD}/${N_FOLDS}"
done

echo "-------------------------------------------------------"
echo "All ${N_FOLDS} jobs submitted. Monitor with: squeue -u \$USER"