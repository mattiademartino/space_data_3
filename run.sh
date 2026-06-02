#!/bin/bash
#SBATCH --job-name=loss_function_ablation2
#SBATCH -n 10
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=11:59:00
#SBATCH --output=slurm_logs/space_data_%j.log

# 2. Activate your virtual environment
source /cluster/home/mriestere/space_data_3/venv/bin/activate

# 3. Run your code
python src/main.py --config src/best_performance_config.yaml
#python src/dino_v2_image_denoiser.py --noisy data/noisy_train_19k_harder.npy --clean data/clean_train_19k_harder.npy