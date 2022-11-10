import skimage.io as io
from skimage.segmentation import mark_boundaries
import numpy as np
import os
import re
import matplotlib.pyplot as plt

import napari
from napari.settings import SETTINGS
SETTINGS.application.ipy_interactive = False


class Correction:
    """
    Functionality of buttons to load, display and correct all 
    images/segmentations of one folder.
    """
    frame = 0

    def __init__(self, ax,
                 im1,
                 path_img,
                 path_seg,
                 files_cut_im,
                 files_seg_im):
        self.im1 = im1
        self.ax = ax
        self.path_img = path_img
        self.path_seg = path_seg
        self.files_cut_im = files_cut_im
        self.files_seg_im = files_seg_im

    def load_img_seg(self):
        """
        Load current image and segmentation.
        """
        self.frame_cut = re.findall(
            '_frame[0-9][0-9][0-9]_', self.files_cut_im[self.frame])[0]
        self.ix_seg = np.where(
            [self.frame_cut in fs for fs in self.files_seg_im])[0][0]

        self.cut_im = io.imread(self.path_img + '/' +
                                self.files_cut_im[self.frame])
        self.seg_im = io.imread(self.path_seg + '/' +
                                self.files_seg_im[self.ix_seg])

        self.seg_im_bin = np.ma.masked_where(self.seg_im == 0, self.seg_im)

        self.overl = mark_boundaries(self.cut_im, self.seg_im, color=(1, 0, 0))

    def update_fig(self):
        """
        Update figure with data of chosen time frame.
        """
        self.im1.set_data(self.overl)
        self.ax.set_title(str(self.frame_cut))
        plt.draw()

    def open_napari(self):
        """
        Open napari for manual correction.
        """
        self.load_img_seg()
        viewer = napari.Viewer()
        viewer.add_image(self.cut_im)
        label_layer = viewer.add_labels(self.seg_im)
        napari.run()
        self.edited_labels = label_layer.data

    def store_corr_seg(self):
        """
        Override segmentation with corrected segmentation.
        """
        orig_seg_dir = self.path_seg + '/orig_seg/'
        if not os.path.exists(orig_seg_dir):
            os.makedirs(orig_seg_dir)
        io.imsave(orig_seg_dir + self.files_seg_im[self.ix_seg], self.seg_im)
        io.imsave(self.path_seg + '/' +
                  self.files_seg_im[self.ix_seg], self.edited_labels)

    def correct_seg(self, event):
        """
        Open napari for manual correction and store corrected segmentation.
        """
        self.open_napari()
        self.store_corr_seg()

    def next_frame(self, event):
        """
        Load and display data of next time frame.
        """
        self.frame += 1
        self.load_img_seg()
        self.update_fig()

    def prev_frame(self, event):
        """
        Load and display data of previous time frame.
        """
        self.frame -= 1
        self.load_img_seg()
        self.update_fig()
