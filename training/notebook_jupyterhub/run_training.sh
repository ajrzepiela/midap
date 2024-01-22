#!/bin/bash

module purge
module load gcc/8.2.0 python_gpu/3.10.4 hdf5/1.10.1 eth_proxy
source ../../euler/midap/bin/activate

FOLDER='my_weights'
mkdir $FOLDER

sbatch --mem-per-cpu=12g --gpus=1 --gres=gpumem:12g --time=01:00:00 --mail-type=START,END,FAIL --wrap "python ../train.py --n_grid 2 --epochs 10 --save_path $FOLDER data/*raw.tif"

