% add path
addpath(genpath('../SuperSegger'))
addpath(genpath('../data'))

% define folder with images
data_folder = '/cluster/work/sis/ri/oschmanf/ackermann-bacteria-segmentation/data/'
image_folder = [data_folder, '15052019_algMono0.1_20190517_segm/raw_im']
%image_folder = [data_folder, image_folders(1).name, '/']

% list all tif files in folder
tif_files = dir(image_folder);

% load image
im = imread([image_folder,filesep,tif_files(3).name]);

% set constants for segmentation and tracking
CONST = loadConstants ('60XEclb',0);
CONST.trackLoci.numSpots = [5];
CONST.trackOpti.NEIGHBOR_FLAG = true;
CONST.parallel.verbose = 0;

% run only tracking
clean_flag = 1;
startEnd = [3 10]
BatchSuperSeggerOpti(image_folder,1,clean_flag,CONST, startEnd);


