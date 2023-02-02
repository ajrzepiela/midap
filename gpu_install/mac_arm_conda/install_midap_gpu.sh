#!/bin/zsh

# check if we have miniconda
if [[ ! -d ~/miniconda ]]; then
  while true; do
    read -q "yn?Miniconda seems missing, do you want to install it? Y/N "
    case $yn in
        [Yy]* ) wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh && bash miniconda.sh -b -p $HOME/miniconda && rm miniconda.sh; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no. ";;
    esac
  done
fi


# update conda and environment
source ~/miniconda/bin/activate
conda env update -f environment_gpu.yml

# activate environment
conda activate midap_gpu

# instal midap
python3 -m pip install -e ../..
