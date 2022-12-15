import numpy as np

import matplotlib
# backend has to be set before the pyplot import, TkAgg is compatible with most clusters
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets  import RectangleSelector


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

        # interactive cutout of chambers
        corners = self.interactive_cutout(img)
        self.corners_cut = tuple([int(i) for i in corners])

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
