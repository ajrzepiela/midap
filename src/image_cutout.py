import numpy as np
import math
import os
import re

import matplotlib
# backend has to be set before the pyplot import, TkAgg is compatible with most clusters
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector

import skimage.measure as measure
import skimage.io as io
from skimage.registration import phase_cross_correlation
from skimage.filters import threshold_mean
import cv2

from tqdm import tqdm

from utils import get_logger

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = os.environ["__VERBOSE"]
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)

class CutoutImage:
    """
    A class that performs the image cutout for the different channels
    """

    # this logger will be shared by all instances and subclasses
    logger = logger

    def __init__(self, paths, min_x_range=700, max_x_range=1600, min_y_range=500,
                 max_y_range=1600, cutout_mode='interactive'):

        # if paths is just a single string we pack it into a list
        if isinstance(paths, str):
            self.paths = [paths]
        else:
            self.paths = paths

        # read out the variables
        self.min_x_range = min_x_range
        self.max_x_range = max_x_range
        self.min_y_range = min_y_range
        self.max_y_range = max_y_range

        # mode for cutout
        self.cutout_mode = cutout_mode

        # get the file lists
        self.channels = [[os.path.join(channel, f) for f in sorted(os.listdir(channel))] for channel in self.paths]

        # adjust the length such that all channels have the same number of elements
        self.min_frames = np.min([len(channel) for channel in self.channels])
        for i in range(len(self.channels)):
            self.channels[i] = self.channels[i][:self.min_frames]

    def align_two_images(self, src_img, ref_img):
        """
        Calculates the shifts necessary to align to images
        :param src_img: Source image
        :param ref_img: Reference image
        :returns: shifts as vector for the alignment
        """
        # aligns a source image in comparison to a reference image
        return phase_cross_correlation(src_img, ref_img)[0].astype(int)

    def align_all_images(self):
        """
        Calculates the shifts necessary to align all images
        """
        # load 1st image of phase channel
        files = self.channels[0]
        src = self.open_tiff(files[0])
        self.shifts = []
        for i in tqdm(range(1,len(files))):
            ref = self.open_tiff(files[i])
            # align image compared to 1st image
            shift = self.align_two_images(src, ref)
            self.shifts.append(shift)

        # computes the offset to choose depending on shift between images
        self.off = int(math.ceil(np.max(np.abs(self.shifts)) / 10.0)) * 10

    def cut_corners(self, img):
        """
        Given a single aligned image as array, it defines the corners that are used to cut out all images
        :param img: Image to cut as array
        :returns: The corners of the cutout as tuple (left_x, right_x, lower_y, upper_y), where full range of the
                  image, i.e. the limits of the corners, are given by the total number of pixels.
        """
        if self.cutout_mode == 'automatic':
            # find contours in image
            contours = self.find_contour(img)
            # compute the range in x-direction for every contour and check for contours above a specific size
            ix_rect = self.find_rectangle(contours)
            # get the corners of the rectangle
            corners = self.get_corners(contours[ix_rect])

            # get coordinates from cutout and cutout images
            self.rectangle_x, self.rectangle_y, self.range_x, self.range_y = self.draw_rectangle(corners)
            self.corners_cut = self.get_corners(np.array([self.rectangle_y, self.rectangle_x]).T)

        elif self.cutout_mode == 'interactive':
            # interactive cutout of chambers
            corners = self.interactive_cutout(img)
            self.corners_cut = tuple([int(i) for i in corners])

    def save_corners(self):
        """
        Saves the corners for the cutout into the settings.sh
        """
        # write the corners into the settings.sh file
        with open("settings.sh", mode="r+") as f:
            # read the file content
            content = f.read()

            if "CORNERS=" in content:
                # we replace the variable
                content = re.sub(f"CORNERS\=.*", f"CORNERS={self.corners_cut}".replace(",", ""), content)

                # truncate, set stream to start and write
                f.truncate(0)
                f.seek(0)
                f.write(content)
            else:
                # we add a new line to the file
                # replace commas to create a bash array
                f.write(f"CORNERS={self.corners_cut}\n".replace(",", ""))

    def open_tiff(self, path):
        """
        Opens a tiff file and returns the images as array
        :param path: path to the file
        :returns: array of images
        """
        # load all pages of tiff file and return list of image arrays
        im = io.imread(path)
        return im

    def find_contour(self, img):
        """
        Finds the contours of an image
        :param img: Image as array
        :returns: The contours of the image calculates with skimage.measure
        """
 
        # binarize image
        thresh = threshold_mean(img)
        img_bin = np.zeros(img.shape)
        img_bin[img > thresh] = 1
        # define kernel
        kernel_vert = np.ones((50,1), np.uint8)
        kernel_hor = np.ones((1,50), np.uint8)
        # dilate image
        d_vert = cv2.dilate(img_bin, kernel_vert, iterations=1)
        d_hor = cv2.dilate(d_vert, kernel_hor, iterations=1)
        # find contours in binary image
        contours = measure.find_contours(d_hor, 0, fully_connected='high')
        return contours

    def find_rectangle(self, contours):
        """
        Tries to find the rectacle of the box from the contours of an image
        :param contours: The contours of the image returned from the <find_contour> function
        :returns: The rectangle that was found from the contours
        """

        # select only those contours which contain a closed shape
        closed_shape_ix = np.where([(c[:,1][0] == c[:,1][-1])&(c[:,0][0] == c[:,0][-1])\
                                                        for c in contours])[0]
        closed_shapes = [contours[ix] for ix in closed_shape_ix]

        #find the rectangular contour
        x_range_contours = np.array([np.max(c[:,1]) - np.min(c[:,1]) for c in closed_shapes])
        y_range_contours = np.array([np.max(c[:,0]) - np.min(c[:,0]) for c in closed_shapes])

        #assume chamber size is above a specific value
        ix = list(set(np.where(x_range_contours > self.min_x_range)[0])&\
        set(np.where(x_range_contours < self.max_x_range)[0])&\
        set(np.where(y_range_contours > self.min_y_range)[0])&\
        set(np.where(y_range_contours < self.max_y_range)[0]))
        ix_rect = closed_shape_ix[ix[0]]
        return ix_rect

    def get_corners(self, shape):
        """
        Returns the bounding box of a shape
        :param shape: A shape as returned from the <find_rectangle> function
        :returns: The corners of the bounding box for that shape
        """
        # returns the corners of the rectangle
        left_x = int(np.min(shape[:,1]))
        right_x = int(np.max(shape[:,1]))
        lower_y = int(np.min(shape[:,0]))
        upper_y = int(np.max(shape[:,0]))
        return left_x, right_x, lower_y, upper_y

    def interactive_cutout(self, img):
        """
        Generates an interactive plot to select the borders of the chamber
        :param img: The image for the plot as array
        :returns: The corners as (left_x, right_x, lower_y, upper_y)
        """
        fig, self.ax = plt.subplots()
        self.ax.imshow(img)

        rs = RectangleSelector(self.ax, self.line_select_callback,
                       drawtype='box', useblit=False, button=[1],
                       minspanx=5, minspany=5, spancoords='pixels',
                       interactive=True)
        plt.show()

        left_x, right_x = rs.corners[0][:2]
        lower_y, upper_y = rs.corners[1][1:3]

        plt.imshow(img)
        plt.xlim([left_x, right_x])
        plt.ylim([lower_y, upper_y])
        plt.show()

        return left_x, right_x, lower_y, upper_y

    def line_select_callback(self, eclick, erelease):
        """
        Line select callback for the RectangleSelector of the <interactive_cutout> routine
        :param eclick: Press event of the mouse
        :param erelease: Release event of the mouse
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
        self.ax.add_patch(rect)

    def draw_rectangle(self, corners, range_x = False, range_y = False, first_image = True):
        # TODO: This function needs a proper Docstring, but I don't know what the arguments do
        # draw a rectangle based on the corners
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
            lower_y = upper_y - range_y

            rectangle_x = [left_x, right_x, right_x, left_x, left_x]
            rectangle_y = [lower_y, lower_y, upper_y, upper_y, lower_y]
            return rectangle_x, rectangle_y

    def do_cutout(self, img, corners_cut):
        """
        Performs a cutout of an image
        :param img: Image ad array
        :param corners_cut: The corners used for the cutout
        :returns: The cutout from the image given the corners
        """
        # generate cutout of image
        left_x, right_x, lower_y, upper_y = corners_cut
        cutout = img[lower_y:upper_y, left_x:right_x]
        return cutout

    def scale_pixel_val(self, img):
        """
        Rescale the pixel values of the image
        :param img: The input image as array
        :returns: The images with pixels scales to standard RGB values
        """
        img_scaled = (255 * ((img - np.min(img))/np.max(img - np.min(img)))).astype('uint8')
        return img_scaled

    def shift_image(self, im, shift):
        """
        Aligns an image of an additional channels by shift
        :param im: image to shift as array
        :param shift: The shift to perform
        :returns: The shifted image
        """
        if self.off-shift[0] != 0:
            im = im[(self.off - shift[0]):(-self.off - shift[0]), :]
        if self.off - shift[1] != 0:
            im = im[:, (self.off - shift[1]):(-self.off - shift[1])]
        return im

    def save_cutout(self, files, file_names):
        """
        Saves the cutouts into the proper directory
        :param files: A list of arrrays (the cutouts) to save
        :param file_names: The list of file names from the original files
        """
        # save of cutouts
        dir_name = os.path.dirname(os.path.dirname(file_names[0]))
        for f, i in zip(file_names, files):
            fname = f"{os.path.splitext(os.path.basename(f))[0]}_cut.png"
            io.imsave(os.path.join(dir_name, 'cut_im', fname), i, check_contrast=False)
        
    def run_align_cutout(self):
        """
        Aligns and cut out all images from all channels
        """

        self.logger.info('Aligning images...')
        self.align_all_images()
        
        self.logger.info('Cutting images...')
        # cycle through the channels
        for channel_id, files in enumerate(self.channels):
            self.logger.info(f'Starting with channel {channel_id+1}/{len(self.channels)}')
            # list for the aligned cutouts
            aligned_cutouts = []

            # get the first image
            src = self.open_tiff(files[0])
            # offset of 1st image
            if self.off == 0:
                src_off = src
            else:
                src_off = src[self.off:-self.off, self.off:-self.off]

            # We get the corners using the PH channel
            if channel_id == 0:
                # set the corner to cut
                self.cut_corners(img=src_off)

                # save the corner to file
                self.save_corners()

            # perform the cutout of the first image
            cutout = self.do_cutout(src_off, self.corners_cut)
            # scale the pixel values
            cut_src = self.scale_pixel_val(cutout)

            # add to list
            aligned_cutouts.append(cut_src)

            # cutout of all other images of all channels
            for i in tqdm(range(1, len(files))):
                ref = self.open_tiff(files[i])

                # align and cutout phase image compared to 1st image
                aligned_img = self.shift_image(ref, self.shifts[i-1])
                cut_img = self.do_cutout(aligned_img, self.corners_cut)
                # sacle the pixel values
                proc_img = self.scale_pixel_val(cut_img)
                aligned_cutouts.append(proc_img)

            self.save_cutout(aligned_cutouts, files)
