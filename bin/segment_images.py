import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm

from PIL import Image, ImageSequence
from scipy.ndimage.morphology import binary_dilation, grey_dilation
import skimage.measure as measure
import cv2

import sys
sys.path.append('../src')
from image_segmentation import *

# load all pages of tiff-file (add argparse)

tiff_dir = '../data/TimeLapse_20190517_15052019_L-algMono-0.1_tiff/'
path = tiff_dir + os.listdir(tiff_dir)[16]

cutouts = cutout_all_pages(path) 
                              
filename = tiff_dir + os.path.splitext(os.path.basename(path))[0] + '.tiff'
imageio.mimwrite(filename,np.array(cutputs))

