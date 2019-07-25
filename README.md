## Project outline
1. Convert of .vsi to .tiff (Done)
2. Automatic detection of chambers (Done for single tiff-file, image "cleaning" is missing)
3. Generation of segmentation images with scikit-image
4. Improvement of segmentation images 
5. Train U-Net
6. Include segmentation-generation into workflow (workflow manager/bash script)

## Starting the virtual environment on Euler
module load new eth_proxy gcc/4.8.2 java/1.8.0_91 python/3.6.1
source ackermann/bin/activate

## Convert .vsi to .tiff
python util/convert_files.py --directory name_of_directory
