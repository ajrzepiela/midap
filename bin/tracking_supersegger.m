function execute = tracking_supersegger(path)
	% add path
	addpath(genpath('../SuperSegger'))
	addpath(genpath(path))
        %addpath(genpath('../data'))

	% define folder with images
	image_folder = path

	% list all tif files in folder
	tif_files = dir(image_folder);
        %fprintf(tif_files)
	% set constants for segmentation and tracking
	CONST = loadConstants ('60XEclb',0); %60XEclb '100XPa'
	CONST.trackLoci.numSpots = [5];
	CONST.trackOpti.NEIGHBOR_FLAG = true;
	CONST.parallel.verbose = 0;

	% run only tracking
	clean_flag = 1; %0 
	startEnd = [3 10] %[3 10]
	BatchSuperSeggerOpti(image_folder,1,clean_flag,CONST,startEnd,0);
	execute = 0;
	exit;
end

