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

from .base_cutout import CutoutImage
from ..utils import get_logger

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = os.environ["__VERBOSE"]
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)

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
