#!/bin/bash
#SBATCH -n 10
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20g
#SBATCH --time=01:10:00

python src/main.py 