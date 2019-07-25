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
all_pages = open_tiff(path)

all_cutouts = []

pbar_pages = tqdm.tqdm(all_pages)
for i, img in enumerate(pbar_pages):
	# scale pixel-values
	img = scale_pixel_vals(img)
	# find contours
	dilation = grey_dilation(img, size=(20,20))
	contours = measure.find_contours(dilation, 200)
	# find rectangle of specific size
	ix_rect = find_rectangle(contours)
	# get the corners of the rectangle
	corners = get_corners(contours[ix_rect])
	if i == 0:
		# scale pixel-values
		#img = scale_pixel_vals(img)
		# find contours 
		#dilation = grey_dilation(img, size=(20,20))
		#contours = measure.find_contours(dilation, 200)
		# find rectangle of specific size
		#ix_rect = find_rectangle(contours)
		# get the corners of the rectangle
		#corners = get_corners(contours[ix_rect])
		rectangle_x, rectangle_y, range_x, range_y = draw_rectangle(corners)
		# generate cutout
		corners_cut = get_corners(np.array([rectangle_y, rectangle_x]).T)
		cutout = do_cutout(img, corners_cut)
		all_cutouts.append(cutout)
	elif i > 0:
                # scale pixel-values
                # img = scale_pixel_vals(img)
                # find contours
                # dilation = grey_dilation(img, size=(20,20))
                # contours = measure.find_contours(dilation, 200)
                # find rectangle of specific size
                # ix_rect = find_rectangle(contours)
		rectangle_x, rectangle_y = draw_rectangle(corners, range_x, 
                                          		  range_y, first_image = False)
		corners_cut = get_corners(np.array([rectangle_y, rectangle_x]).T)
		cutout = do_cutout(img, corners_cut)
		all_cutouts.append(cutout)

filename = tiff_dir + os.path.splitext(os.path.basename(path))[0] + '.tiff'
imageio.mimwrite(filename,np.array(images))

