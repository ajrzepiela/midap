#!/bin/bash

# set the path to the installation
PATH_MINIFORGE=$HOME'/miniforge3'

# miniforge installation and env creation
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

chmod +x Miniforge3-MacOSX-arm64.sh

./Miniforge3-MacOSX-arm64.sh -b -p $PATH_MINIFORGE

$PATH_MINIFORGE/bin/conda env create -f environment_m1.yml

# instal midap
${HOME}/miniforge3/envs/workflow/bin/python3 -m pip install -e .

#echo alias workflow_m1=$PATH_MINIFORGE'/envs/workflow/bin/python3.8' >> $HOME/.zshrc
#source $HOME/.zshrc
