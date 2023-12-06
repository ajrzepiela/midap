<table><tr><td valign="center"> 
  <img align="left" height="25px" src="https://github.com/Microbial-Systems-Ecology/midap/actions/workflows/pytest_with_conda.yml/badge.svg?branch=development">
  <img align="left" height="25px" src="https://github.com/Microbial-Systems-Ecology/midap/actions/workflows/pytest_with_venv.yml/badge.svg?branch=development">
  <img align="left" height="25px" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jafluri/9219639a376674762e7e29e2fa3cfc9e/raw/midap_coverage.json">
  <b> (Development Branch) </b>
</td></tr></table>

# MIDAP

## Standard Installation

The installation was tested on macOS Big Sur (11.6.7), Ubuntu 22.04 and WSL II on Win 11. Note that the standard installation does **not** support GPUs. If you want to run the pipeline with GPU support, please have a look at the relevant sections below.

1. For the download, either:

- Clone the repo, navigate to the directory containing the pipeline `cd midap`.
- Download the [latest release](https://github.com/Microbial-Systems-Ecology/midap/releases) and unpack the tar.gz-file. Then navigate to the unpacked directory using the command line `cd midap-VERSION` (and add your version number).

2. Create and activate the conda environment:

```
conda create --name midap python=3.10
conda activate midap
pip install -e .
```

3. Once the conda environment is activated, you can run the module from anywhere via `midap`. If you run the pipeline for the first time, it will download all the required files (~3 GB). You can also manually (re)download the files using the command `midap_download`. The module accepts arguments and has the following signature:

```
usage: midap [-h] [--restart [RESTART]] [--headless] [--loglevel LOGLEVEL] [--cpu_only] [--create_config]

Runs the cell segmentation and tracking pipeline.

optional arguments:
  -h, --help           show this help message and exit
  --restart [RESTART]  Restart pipeline from log file. If a path is specified the checkpoint and settings file will be
                       restored from the path, otherwise the current working directory is searched.
  --headless           Run pipeline in headless mode, ALL parameters have to be set prior in a config file.
  --loglevel LOGLEVEL  Set logging level of script (0-7), defaults to 7 (max log)
  --cpu_only           Sets CUDA_VISIBLE_DEVICES to -1 which will cause most! applications to use CPU only.
  --create_config      If this flag is set, all other arguments will be ignored and a 'settings.ini' config file is
                       generated in the current working directory. This option is meant generate config file templates
                       for the '--headless' mode. Note that this will overwrite if a file already exists.
```

For an installation with GPU support, please refer to the documentation.   

## Documentation

The documentation can be found in the [wiki](https://github.com/Microbial-Systems-Ecology/midap/wiki).

## Issues

If you are having trouble with the package, please have a look at the [troubleshooting page](https://github.com/Microbial-Systems-Ecology/midap/wiki/Troubleshooting#creating-an-github-issue) 
and if necessary create an issue according to the instructions provided there.  
