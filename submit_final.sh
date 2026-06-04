#!/bin/bash
# Submit both final training arrays in one shot.
# Usage:  bash submit_final.sh

set -e

echo "Submitting models 1-8 (unet_deep, unet_attention, resnet) — 3h each..."
JOB1=$(sbatch --parsable run_final.sh)
echo "  Job array ID: $JOB1"

echo "Submitting models 9-10 (diffusion) — 12h each..."
JOB2=$(sbatch --parsable run_final_diffusion.sh)
echo "  Job array ID: $JOB2"

echo ""
echo "All 10 jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "When all jobs finish, generate the summary:"
echo "  python src/train_final.py --summarize"
