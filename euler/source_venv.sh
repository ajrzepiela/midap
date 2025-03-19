#!/bin/bash

# actiavte the right modules
module load stack/2024-06 python_cuda/3.11.6 cudnn/8.9.7.29-12 eth_proxy || true

# activate the env, use realpath and dir name to make it sourable from everywhere
source $(realpath $BASH_SOURCE | xargs dirname)/midap/bin/activate

#point to correct cuda version
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT
