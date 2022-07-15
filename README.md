# MIDAP

## Installation on Mac and Linux

The installation was tested on macOS Big Sur (11.6.7) and Ubuntu 22.04.

1. Download model weights and example files from polybox `./download_files.sh`

2. Create virtual environment `conda env create -f environment.yml`

3. Navigate to bin-directory and activate conda environment
```
cd bin/
conda activate workflow
```
4. Start pipeline from the command line with `./run_pipeline_test.sh`

## Installation on new MacBooks with M1

1. Install Miniforge `./install_miniforge.sh`

2. Navigate to bin-directory and run the pipeline:
```
cd bin/
./run_pipeline_m1.sh
```

## User Guide

### Pipeline
1. Start pipeline as described above.

2. Select the data type<br/>
![Screenshot_1](img/window_select.png)<br/>

3. In case of family or mother machine, select the part of the pipeline you want to run, the frame numbers to restrict the analysis to, the folder path, identifiers of input files, identifiers of position/experiment and whether you want to deconvolve the raw images.
![Screenshot_1](img/window_chamber_new.png)<br/>
The pipeline requires grayscale tiff-stacks as input files.
If only one or two out of three channels were used, specify only the identifiers of those channels and leave the other fields free. By default the phase images are not segmented. In case you would like to do cell segmentation and tracking for your phase images, place click the box next to the respective field.
**Please note:** all input fields are case sensitive!
4. In case of well plates, select the part of the pipeline you want to run, the frame numbers to restrict the analysis to, the file name and whether you want to deconvolve the raw images.
<br/>
![Screenshot_1](img/window_well.png)<br/>

### Manual correction of segmentations
The manual correction can be started with the following commands:
```
cd bin/
python correct_segmentation.py --path_img PATH_IMG --path_seg PATH_SEG_IMG
```

The arguments PATH_IMG and PATH_SEG_IMG are passed as strings and should contain the full path name (e.g. '/Users/Documents/data/img_1.tif').

### Visualization of tracking results
```
cd bin/
python visualize_lineages.py --path ../example_data/Glen/{Position}/{Channel}/track_output/
```

