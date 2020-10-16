### Setup on local machine (Mac)
#### 1. Initialization and update of submodule (SuperSegger)
```
git submodule init
git submodule update
```

#### 2. Setup of environment
```
conda env create -f environment_mac.yml
```

#### 3. Navigate to bin-directory and mark bash-script as executable
```
cd bin/
chmod +x run_pipeline_local.sh
```

### Run pipeline on local machine
#### 1. Navigate to bin-directory and activate conda environment
```
cd bin/
conda activate workflow
```

#### 2. Execute pipeline
```
./run_pipeline_local.sh
```

#### 3. Select data type, folder path and identifiers of input files via GUI

1. Select the data type<br/>
![Screenshot_1](img/window_select.png)<br/>

2. In case of family or mother machine, select the part of the pipeline you want to run, the frame numbers to restrict the analysis to, the folder path, identifiers of input files, whether you want to deconvolve the raw images and the Matlab root folder.
![Screenshot_1](img/window_chamber.png)<br/>
If only one or two out of three channels were used, specify only the identifiers of those channels and leave the other fields free. 

3. In case of well plates, select the part of the pipeline you want to run, the frame numbers to restrict the analysis to, the file name, whether you want to deconvolve the raw images and the Matlab root folder.
![Screenshot_1](img/window_well.png)<br/>

3. The matlab root folder can be found with the following Matlab-command:
```
matlabroot
```
