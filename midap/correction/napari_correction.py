"""
Correct segmentations
===============================

Display overlay of all images/segmentations of a folder and open correction if needed.
"""

import os
import re
from typing import List

import numpy as np
from skimage import io
from skimage.segmentation import mark_boundaries

import matplotlib.axes
import matplotlib.image
import matplotlib.backend_bases
import matplotlib.pyplot as plt

import napari
from napari.settings import SETTINGS
SETTINGS.application.ipy_interactive = False


class Correction:
    '''
    Functionality of buttons to load, display and correct all 
    images/segmentations of one folder.
    :param frame: frame ID of first frame
    :param cur_frame: current frame ID
    :param ix_seg: index of corresponding seg file
    :param cut_im: cut-out image
    :param seg_im: segmentation image
    :param overl: overlay of cut-out and segmentation
    :param ax: axis of displayed figure
    :param path_img: path to images folder
    :param path_seg: path to segmentations folder
    :param files_cut_im: list with file names of images
    :param files_seg_im: list with file names of segmentations
    '''
    frame: int = 0
    cur_frame: int
    ix_seg: int
    cut_im: int
    seg_im: int
    overl: int

    def __init__(self, ax: matplotlib.axes.Axes,
                 path_img: str,
                 path_seg: str,
                 files_cut_im: List[str],
                 files_seg_im: List[str]) -> None:
        self.ax = ax
        self.path_img = path_img
        self.path_seg = path_seg
        self.files_cut_im = files_cut_im
        self.files_seg_im = files_seg_im

    def load_img_seg(self, frame: int) -> None:
        '''
        Load current image and segmentation and generate overlay.
        :param frame: current file index
        '''
        self.cur_frame = re.findall(
            '_frame[0-9][0-9][0-9]_', self.files_cut_im[frame])[0]
        self.ix_seg = np.where(
            [self.cur_frame in fs for fs in self.files_seg_im])[0][0]

        self.cut_im = io.imread(os.path.join(self.path_img, self.files_cut_im[self.frame]))
        self.seg_im = io.imread(os.path.join(self.path_seg, self.files_seg_im[self.ix_seg]))
        self.orig_seg_im = self.seg_im.copy()

        self.overl = mark_boundaries(self.cut_im, self.seg_im, color=(1, 0, 0))

    def update_fig(self, im: matplotlib.image.AxesImage) -> None:
        '''
        Update figure with data of chosen time frame.
        :param im: image object used to update figure
        '''
        im.set_data(self.overl)
        self.ax.set_title(str(self.cur_frame))
        plt.draw()

    def open_napari(self) -> None:
        '''
        Open napari for manual correction.
        '''
        self.load_img_seg(self.frame)
        viewer = napari.Viewer()
        viewer.add_image(self.cut_im)
        label_layer = viewer.add_labels(self.seg_im)
        napari.run()
        self.edited_labels = label_layer.data

    def store_corr_seg(self) -> None:
        '''
        Override segmentation with corrected segmentation.
        '''
        orig_seg_dir = os.path.join(self.path_seg, 'orig_seg')
        os.makedirs(orig_seg_dir, exist_ok=True)
        io.imsave(os.path.join(orig_seg_dir, self.files_seg_im[self.ix_seg]), self.orig_seg_im)
        io.imsave(os.path.join(self.path_seg,self.files_seg_im[self.ix_seg]), self.edited_labels)

    def correct_seg(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        '''
        Open napari for manual correction and store corrected segmentation.
        :param event: mouse event for "Correction" button 
        '''
        self.open_napari()
        self.store_corr_seg()

    def next_frame(self, event: matplotlib.backend_bases.MouseEvent, im: matplotlib.image.AxesImage) -> None:
        '''
        Load and display data of next time frame.
        :param event: mouse event for "Next" button
        :param im: image object used to update figure
        '''
        self.frame += 1
        self.load_img_seg(self.frame)
        self.update_fig(im)

    def prev_frame(self, event: matplotlib.backend_bases.MouseEvent, im: matplotlib.image.AxesImage) -> None:
        '''
        Load and display data of previous time frame.
        :param event: mouse event for "Previous" button 
        :param im: image object used to update figure
        '''
        self.frame -= 1
        self.load_img_seg(self.frame)
        self.update_fig(im)
