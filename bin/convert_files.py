import os
import glob
import numpy as np

import argparse
import bioformats as bf
import imageio
import javabridge as jb
import tqdm

import sys
sys.path.append('../src')
from file_conversion import *

# pass argument
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str, help="Name of .vsi file to convert.")
ap.add_argument("-td", "--tiff_dir", type=str, help="Name of directory to store the converted tiff file.")
args = vars(ap.parse_args())

#directory = args['directory'] #name of directory with vsi-files:
#vsi_dir, tiff_dir = create_dir_name(directory)
#create_dir(tiff_dir)

#vsi_files = glob.glob(vsi_dir + '/*.vsi')
do_convert(args['file'], args['tiff_dir'])
	

