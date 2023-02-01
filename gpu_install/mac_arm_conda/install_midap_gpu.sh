#!/bin/bash


# update conda and environment
conda env update -f environment_gpu.yml

# activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate midap_gpu

# instal midap
python3 -m pip install -e ../..
