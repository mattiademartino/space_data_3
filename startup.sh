#!/bin/bash
module purge

module load stack/2024-06  gcc/12.2.0  cuda/12.1.1  python/3.10.13  eth_proxy
#pip uninstall -y torch torchvision numpy "nvidia-*" triton || true

# pip install --user "numpy<2" \
#   torch==2.2.1+cu121  torchvision==0.17.1+cu121 \
#   matplotlib  pytorch_msssim \
#   --extra-index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -r requirements.txt
source ./venv/bin/activate