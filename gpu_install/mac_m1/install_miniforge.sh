#!/bin/bash

# set the path to the installation
PATH_MINIFORGE=$HOME'/miniforge3'

# install miniforge if not alreade installed
if [ ! -d "$PATH_MINIFORGE" ]; then
  # miniforge installation and env creation
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
  
  chmod +x Miniforge3-MacOSX-arm64.sh
  
  ./Miniforge3-MacOSX-arm64.sh -b -p $PATH_MINIFORGE
fi

# update conda and environment
$PATH_MINIFORGE/bin/conda update -n base -c conda-forge conda
$PATH_MINIFORGE/bin/conda env update -f environment_m1.yml

# instal midap
${HOME}/miniforge3/envs/midap_gpu/bin/python3 -m pip install -e ../..
