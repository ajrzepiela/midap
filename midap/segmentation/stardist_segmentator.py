import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from stardist.models import StarDist2D
from csbdeep.utils import normalize

from .base_segmentator import SegmentationPredictor
from ..utils import GUI_selector


class StarDistSegmentation(SegmentationPredictor):
    """
    A class that performs the image segmentation of the cells using a UNet
    """

    supported_setups = ["Family_Machine", "Mother_Machine"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the UNetSegmentation using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """

        # base class init
        super().__init__(*args, **kwargs)

    def set_segmentation_method(self, path_to_cutouts):
        """
        Performs the weight selection for the segmentation network. A custom method should use this function to set
        self.segmentation_method to a function that takes an input images and returns a segmentation of the image,
        i.e. an array in the same shape but with values only 0 (no cell) and 1 (cell)
        :param path_to_cutouts: The directory in which all the cutout images are
        """

        if self.model_weights is None:
            self.logger.info('Selecting weights...')

            # get the image that is roughly in the middle of the stack
            list_files = np.sort(os.listdir(path_to_cutouts))
            # take the middle image (but round up, if there are only 2 we want the second)
            if len(list_files) == 1:
                ix_half = 0
            else:
                ix_half = int(np.ceil(len(list_files) / 2))

            path_img = list_files[ix_half]

            # scale the image and pad
            img = self.scale_pixel_vals(io.imread(os.path.join(path_to_cutouts, path_img)))
            self.logger.info(f'The shape of the image is: {img.shape}')

            # display different segmentation models
            labels = ['2D_versatile_fluo', '2D_paper_dsb2018', '2D_versatile_he']
            figures = []
            for model_name in labels:
                model = StarDist2D.from_pretrained('2D_versatile_fluo')
                # predict, we only need the mask, see omnipose tutorial for the rest of the args
                mask, _ = model.predict_instances(normalize(img))
                # omni removes axes that are just 1
                seg = (mask > 0.5).astype(int)

                # now we create a plot that can be used as a button image
                fig, ax = plt.subplots(figsize=(3,3))
                ax.imshow(img)
                ax.contour(seg, [0.5], colors='r', linewidths=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(model_name)
                figures.append(fig)

            # Title for the GUI
            channel = os.path.basename(os.path.dirname(path_to_cutouts))
            # if we just got the chamber folder, we need to go one more up
            if channel.startswith('chamber'):
                channel = os.path.basename(os.path.dirname(os.path.dirname(path_to_cutouts)))
            title = f'Segmentation Selection for channel: {channel}'

            # start the gui
            marked = GUI_selector(figures=figures, labels=labels, title=title)

            # set weights
            self.model_weights = marked

        # helper function for the seg method
        model = StarDist2D.from_pretrained(self.model_weights)

        def seg_method(imgs):
            masks = []
            for img in imgs:
                img = self.scale_pixel_vals(img)
                mask, _ = model.predict_instances(normalize(img))
                masks.append(mask)
            # add the channel dimension and batch if it was 1
            return np.stack(masks, axis=0)

        # set the segmentations method
        self.segmentation_method = seg_method
