import numpy as np
from PIL import Image, ImageSequence

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

def find_rectangle(contours):
	#find the rectangular contour 
	x_range_contours = np.array([np.max(c[:,1]) - np.min(c[:,1]) for c in contours])
        #assume chamber size is above a specific value
	ix = np.where(x_range_contours > 500)[0]
	ix_rect = ix[np.where([contours[i][:,1][0] == contours[i][:,1][-1] for i in ix])[0][0]]
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
		upper_y = lower_y + range_y

		rectangle_x = [left_x, right_x, right_x, left_x, left_x]
		rectangle_y = [lower_y, lower_y, upper_y, upper_y, lower_y]
		return rectangle_x, rectangle_y

def do_cutout(img, corners_cut):
	left_x, right_x, lower_y, upper_y = corners_cut
	cutout = img[lower_y:upper_y, left_x:right_x]
	return cutout
