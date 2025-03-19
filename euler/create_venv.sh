#!/bin/bash

# actiavte the right modules
# stack 2024-06 with ggc 12.2.0 python 3.11.6 + cuda and cunn 8.9.7
module load stack/2024-06 python_cuda/3.11.6 cudnn/8.9.7.29-12 eth_proxy || true

# create the env
python -m venv midap

# activate the env
source midap/bin/activate

# install the requirements whithout TF
pip --disable-pip-version-check install -U pip
pip install -r requirements.txt

# install the package
export MIDAP_INSTALL_VERSION=euler
pip install -e .. 

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT

while true; do
    read -p "Do you want to add the source script to your .bash_profile? Y/N " yn
    case $yn in
        [Yy]* ) printf '%s\n' '' '# midap env' "source $(pwd)/source_venv.sh" >> ${HOME}/.bash_profile; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no. ";;
    esac
done
