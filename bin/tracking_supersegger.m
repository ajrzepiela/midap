function execute = tracking_supersegger(path)
	% add path
	addpath(genpath('../SuperSegger'))
	addpath(genpath(path))
    %addpath(genpath('../data'))

	% define folder with images
	image_folder = path

	% set constants for segmentation and tracking
	CONST = loadConstants ('60XEclb',0); %'60XEclb' '100XPa'
	CONST.getLocusTracks.TimeStep = 1;
	CONST.trackOpti.NEIGHBOR_FLAG = true;
	CONST.trackOpti.MIN_CELL_AGE = 3; %1%3
	CONST.trackLoci.numSpots = [0];
	CONST.parallel.verbose = 0;
	CONST.superSeggerOpti.MAX_SEG_NUM = 100000;
	CONST.regionOpti.MAX_NUM_RESOLVE = 100000;
	CONST.findFocusSR.MAX_TRACE_NUM = 100000;

	%CONST.trackOpti.REMOVE_STRAY = true;
	%CONST.trackOpti.MIN_AREA = 30; %0;
	%CONST.trackOpti.MIN_AREA_NO_NEIGH = 30; %0;
	%CONST.trackOpti.MIN_CELL_AGE = 3;
	%CONST.trackOpti.REMOVE_STRAY = true;
	%CONST.trackLoci.numSpots = [5];
	

	% run only tracking
	clean_flag = 1; %0 
	startEnd = [3 10] %[3 10]
	BatchSuperSeggerOpti(image_folder,1,clean_flag,CONST,startEnd,0);
	execute = 0;
	exit;
end

