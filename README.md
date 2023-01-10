# MIDAP

## Installation on Mac and Linux

The installation was tested on macOS Big Sur (11.6.7), Ubuntu 22.04 and WSL II on Win 11.

1. Clone the repo, navigate to the directory containing the pipeline `cd midap` and download model weights and example files from polybox `./download_files.sh`.

2. Create the virtual environment:

    1. **For Macs with an M1 chip:**

       If you are not sure if your Mac has an M1 chip, open a terminal an run
       ```
       if [[ ${OSTYPE} = darwin* ]] && [[ $(uname -m) == "arm64" ]]; then echo "M1"; fi 
       ```
       if it prints "M1" in your terminal, install Miniforge via `./install_miniforge.sh` and proceed to step 3, otherwise proceed with step 2 .

    
    2.  **For Linux and older Macs:**
    
        You can create a conda environment with: `conda env update -f environment.yml` and then activate it via

         ```
         conda activate midap
         ```

3. Once the virtual environment is activated, you can run the module from anywhere via `midap`. The module accepts arguments and has the following signature:

```
usage: midap [-h] [--restart [RESTART]] [--headless] [--loglevel LOGLEVEL] [--cpu_only] [--create_config]

Runs the cell segmentation and tracking pipeline.

optional arguments:
  -h, --help           show this help message and exit
  --restart [RESTART]  Restart pipeline from log file. If a path is specified the checkpoint and settings file will be restored from the path, otherwise the current working directory is
                       searched.
  --headless           Run pipeline in headless mode, ALL parameters have to be set prior in a config file.
  --loglevel LOGLEVEL  Set logging level of script (0-7), defaults to 7 (max log)
  --cpu_only           Sets CUDA_VISIBLE_DEVICES to -1 which will cause most! applications to use CPU only.
  --create_config      If this flag is set, all other arguments will be ignored and a 'settings.ini' config file is generated in the current working directory. This option is meant
                       generate config file templates for the '--headless' mode. Note that this will overwrite if a file already exists.
```

## Installation on the Euler cluster

The installation on Euler is described in the [README.md](./euler/README.md) of the `euler` directory.

## User Guide

### Pipeline
1. Start pipeline as described above.

2. Select the data type<br/>
<img src="img/window_select.png" alt="drawing" width="300"/><br>

3. In case of family or mother machine, please select/add:
- the **part of the pipeline** you want to run
- the **frame numbers** to be analyzed
- **path** to the folder containing the data (tiff-stacks)
- **filetype** of the input files
- **identifiers** of position/experiment
- a comma separated **list of additional channels**
- [optional] **modified methods** for chamber cutout and segmentation (more information: [Modularity](#modularity-of-the-pipeline))
- whether **deconvolution** of the raw images should be applied.<br/>
<img src="img/window_chamber_new.png" alt="drawing" width="500"/><br>
The pipeline requires grayscale tiff-stacks as input files.
By default the phase images are not segmented. In case you would like to do cell segmentation and tracking for your phase images, place click the box next to the respective field.
**Please note:** all input fields are case sensitive!

### Manual correction of segmentations

The scripts for manual correction and visualization use the Python package napari. The scrip can be run via the `correct_segmentation` command which has the signature:

```
usage: correct_segmentation [-h] --path_img PATH_IMG --path_seg PATH_SEG

optional arguments:
  -h, --help           show this help message and exit
  --path_img PATH_IMG  Path to raw image folder.
  --path_seg PATH_SEG  Path to segmentation folder.
```

### Modularity of the Pipeline

It is possible to define custom methods for the chamber cutout, the cell segmentation and the tracking. 

#### Chamber cutout

To define a custom method for the chamber cutout, you can start by copying the `interactive_cutout.py` file in the `midap` package:

```
cd midap/imcut
cp interactive_cutout.py <your_filename>.py
```
In the copied file, change the name of the class from `InteractiveCutout` to your own class. Choose a descriptive name as the name of this class will be shown in the dropdown menu of the GUI to select the method. Then you can overwrite the `cut_corners` method with your own method. Note that you should not add additional arguments to the method and the method has to set the attribute `self.corners_cut` the cutout corners.

#### Cell Segmentation

To define a custom method for the cell segmentation, you can start by copying the `unet_segmentator.py` file in the `midap` package:

```
cd midap/segmentation
cp unet_segmentator.py <your_filename>.py
```
In the copied file, change the name of the class from `UNetSegmentation` to your own class. Choose a descriptive name as the name of this class will be shown in the dropdown menu of the GUI to select the method. Then you can overwrite the `set_segmentation_method` method with your own method. Note that you should not add additional arguments to the method and the method has to set the attribute `self.segmentation_method` to a method that performs the cell segmentation.

#### Tracking

To define a custom method for the cell tracking, you can start by copying the `deltav2_tracking.py` file in the `midap` package:

```
cd midap/tracking
cp deltav2_tracking.py <your_filename>.py
```
In the copied file, change the name of the class from `DeltaV2Tracking` to your own class. Choose a descriptive name as the name of this class will be shown in the dropdown menu of the GUI to select the method. Then you can overwrite the `load_model` method with your own method. Note that you should not add additional arguments to the method and the method has to set the attribute `self.model` to a model that performs the cell tracking. This model should be callable like the DeltaV2 model (see the [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009797) for more information). You can make use of all the attributes that the base class (`DeltaTypeTracking` defined in `base_tracking.py`) sets in its constructor.

## Training segmentation models

You can train the standard MIDAP UNet for segmentation from scratch, finetune existing models or train custom UNets and easily add them to the pipeline. For more information please have a look into the [training](./training) directory.
