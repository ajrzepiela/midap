import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm

from PIL import Image, ImageSequence
from scipy.ndimage.morphology import binary_dilation, grey_dilation
import skimage.measure as measure
import cv2

def open_tiff(path):
	# load all pages of tiff file
	im = Image.open(path)
	all_pages = []
	for i, page in enumerate(ImageSequence.Iterator(im)):
    		all_pages.append(np.array(page))
	return all_pages

def scale_pixel_vals(img):
	# scales pixel values for a range between 0 and 255
	img = np.round((img/np.max(img))*255)
	img = np.float32(img)
	return img

def find_rectangle(contours, wind_width, wind_height):	
	# select only those contour which contain a closed shape
	closed_shape_ix = np.where([(c[:,1][0] == c[:,1][-1])&(c[:,0][0] == c[:,0][-1])\
                            for c in contours])[0]
	closed_shapes = [contours[ix] for ix in closed_shape_ix]

	#find the rectangular contour 
	x_range_contours = np.array([np.max(c[:,1]) - np.min(c[:,1]) for c in closed_shapes])
	y_range_contours = np.array([np.max(c[:,0]) - np.min(c[:,0]) for c in closed_shapes])
        #assume chamber size is above a specific value
	ix = list(set(np.where(x_range_contours > wind_width)[0])&set(np.where(y_range_contours > wind_height)[0]))
	#ix_rect = ix[np.where([contours[i][:,1][0] == contours[i][:,1][-1] for i in ix])[0][0]]
	ix_rect = closed_shape_ix[ix[0]]
	return ix_rect

def get_corners(shape):
	# returns the corners of the rectangle
	left_x = int(np.min(shape[:,1]))
	right_x = int(np.max(shape[:,1]))
	lower_y = int(np.min(shape[:,0]))
	upper_y = int(np.max(shape[:,0]))
	return (left_x, right_x, lower_y, upper_y)

def draw_rectangle(corners, range_x = False, range_y = False, first_image = True):
	# draws a rectangle based on the corners
	left_x, right_x, lower_y, upper_y = corners
	if first_image:
		# draws a rectangle based on the corners
		left_x, right_x, lower_y, upper_y = corners
		range_x = right_x - left_x
		range_y = upper_y - lower_y

		rectangle_x = [left_x, right_x, right_x, left_x, left_x]
		rectangle_y = [lower_y, lower_y, upper_y, upper_y, lower_y]
		return rectangle_x, rectangle_y, range_x, range_y

	else:
		right_x = left_x + range_x
		#upper_y = lower_y + range_y
		lower_y = upper_y - range_y
		
		rectangle_x = [left_x, right_x, right_x, left_x, left_x]
		rectangle_y = [lower_y, lower_y, upper_y, upper_y, lower_y]
		return rectangle_x, rectangle_y

def do_cutout(img, corners_cut):
	left_x, right_x, lower_y, upper_y = corners_cut
	cutout = img[lower_y:upper_y, left_x:right_x]
	return cutout

def cutout_all_pages(path, wind_width, wind_height):
	all_pages = open_tiff(path)

	all_cutouts = []
	pbar_pages = tqdm.tqdm(all_pages)
	for i, img in enumerate(pbar_pages):
        	# scale pixel-values
		img = scale_pixel_vals(img)
		# enhance contrast
		img = img.astype('uint8')
		enhanced = cv2.equalizeHist(img)
		# find contours
		#dilation = grey_dilation(img, size=(20,20))
		contours = measure.find_contours(enhanced, 180,  fully_connected='high')
		# find rectangle of specific size
		ix_rect = find_rectangle(contours, wind_width, wind_height)
		# get the corners of the rectangle
		corners = get_corners(contours[ix_rect])
        	
		# rectangle for first image
		if i == 0:
                	rectangle_x, rectangle_y, range_x, range_y = draw_rectangle(corners)
                	# generate cutout
                	corners_cut = get_corners(np.array([rectangle_y, rectangle_x]).T)
                	cutout = do_cutout(img, corners_cut)
                	all_cutouts.append(cutout)
		elif i > 0:
                	rectangle_x, rectangle_y = draw_rectangle(corners, range_x,
                                                          range_y, first_image = False)
                	corners_cut = get_corners(np.array([rectangle_y, rectangle_x]).T)
                	cutout = do_cutout(img, corners_cut)
                	all_cutouts.append(cutout)

	return all_cutouts
