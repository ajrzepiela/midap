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
from skimage.filters import threshold_mean
import cv2

from .base_cutout import CutoutImage

class InteractiveCutout(CutoutImage):
    """
    A class that performs the image cutout for the different channels in interactive mode
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the class with given arguments and keyword arguments
        :*args: arguments used to init the parent class
        :**kwargs: keyword arguments used to init the parent class
        """
        # init the super class
        super().__init__(*args, **kwargs)

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
        kernel_vert = np.ones((50, 1), np.uint8)
        kernel_hor = np.ones((1, 50), np.uint8)
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
        closed_shape_ix = np.where([(c[:, 1][0] == c[:, 1][-1]) & (c[:, 0][0] == c[:, 0][-1]) \
                                    for c in contours])[0]
        closed_shapes = [contours[ix] for ix in closed_shape_ix]

        # find the rectangular contour
        x_range_contours = np.array([np.max(c[:, 1]) - np.min(c[:, 1]) for c in closed_shapes])
        y_range_contours = np.array([np.max(c[:, 0]) - np.min(c[:, 0]) for c in closed_shapes])

        # assume chamber size is above a specific value
        ix = list(set(np.where(x_range_contours > self.min_x_range)[0]) & \
                  set(np.where(x_range_contours < self.max_x_range)[0]) & \
                  set(np.where(y_range_contours > self.min_y_range)[0]) & \
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
        left_x = int(np.min(shape[:, 1]))
        right_x = int(np.max(shape[:, 1]))
        lower_y = int(np.min(shape[:, 0]))
        upper_y = int(np.max(shape[:, 0]))
        return left_x, right_x, lower_y, upper_y
