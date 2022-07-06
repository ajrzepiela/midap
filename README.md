# 1. Pipeline set-up

## 1. Download model weights and example data from polybox
```
chmod +x download_files.sh
./download_files.sh
```

## 2. Prepare virtual environment on MacBooks (instruction for users of MacBook M1 can be found below.)
#### Install conda environment
```
conda env create -f environment_mac.yml
```

## 3. Run pipeline on local machine
#### Navigate to bin-directory and activate conda environment
```
cd bin/
conda activate workflow
```

#### Execute pipeline
```
./run_pipeline_test.sh
```

## In case you use a new MacBook M1, please follow the following instructions to set up the pipeline:
## 1. Install Miniforge
```
chmod +x install_miniforge.sh
./install_miniforge.sh
```
## 2. Run pipeline
Navigate to bin-directory and mark bash-script as executable. Then run the pipeline:
```
cd bin/
chmod +x run_pipeline_m1.sh
./run_pipeline_m1.sh
```

# 2. Run pipeline
#### Select data type, folder path and identifiers of input files via GUI

1. Select the data type<br/>
![Screenshot_1](img/window_select.png)<br/>

2. In case of family or mother machine, select the part of the pipeline you want to run, the frame numbers to restrict the analysis to, the folder path, identifiers of input files, identifiers of position/experiment and whether you want to deconvolve the raw images.
![Screenshot_1](img/window_chamber_new.png)<br/>
The pipeline requires grayscale tiff-stacks as input files.
If only one or two out of three channels were used, specify only the identifiers of those channels and leave the other fields free. By default the phase images are not segmented. In case you would like to do cell segmentation and tracking for your phase images, place click the box next to the respective field.
**Please note:** all input fields are case sensitive!
3. In case of well plates, select the part of the pipeline you want to run, the frame numbers to restrict the analysis to, the file name and whether you want to deconvolve the raw images.
![Screenshot_1](img/window_well.png)<br/>

# 3. Manual correction of segmentations
The manual correction can be started with the following commands:
```
cd bin/
python correct_segmentation.py --path_img PATH_IMG --path_seg PATH_SEG_IMG
```

The arguments PATH_IMG and PATH_SEG_IMG are passed as strings and should contain the full path name (e.g. '/Users/Documents/data/img_1.tif').

# 4. Visualization of tracking results
```
cd bin/
python visualize_lineages.py --path ../example_data/Glen/{Position}/{Channel}/track_output/
```

