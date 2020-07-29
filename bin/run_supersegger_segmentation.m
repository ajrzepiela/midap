function execute = run_supersegger_segmentation(path)
	% add path
	addpath(genpath('../SuperSegger'))
	addpath(genpath('../data'))

	% define folder with images
	image_folder = path

	% list all tif files in folder
	tif_files = dir(image_folder);

	% set constants for segmentation and tracking
	CONST = loadConstants ('100XPa',0); %60XEclb
	CONST.trackLoci.numSpots = [5];
	CONST.trackOpti.NEIGHBOR_FLAG = true;
	CONST.parallel.verbose = 0;

	% run only tracking
	clean_flag = 1; %0 
	%startEnd = [2 3] %[3 10]
	BatchSuperSeggerOpti(image_folder,1,clean_flag,CONST);%, startEnd,0);
	execute = 0;
	exit;
end

