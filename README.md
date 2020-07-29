### Setup on local machine (Mac)
#### 1. Initialization and update of submodule (SuperSeggar)
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
#### 1. Navigate to bin-directory and execute pipeline
```
cd bin/
./run_pipeline_local.sh
```

#### 2. Select folder path and input files via GUI

