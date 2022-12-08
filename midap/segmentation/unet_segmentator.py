import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.widgets import RadioButtons

from ..networks.unets import UNetv1
from .base_segmentator import SegmentationPredictor


class UNetSegmentation(SegmentationPredictor):
    """
    A class that performs the image segmentation of the cells using a UNet
    """

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

        self.logger.info('Selecting weights...')

        # get the image that is roughly in the middle of the stack
        list_files = np.sort(os.listdir(path_to_cutouts))
        ix_half = int(len(list_files) / 2)
        path_img = list_files[ix_half]

        # scale the image and pad
        img = self.scale_pixel_vals(io.imread(os.path.join(path_to_cutouts, path_img)))
        img_pad = self.pad_image(img)

        # try watershed segmentation as classical segmentation method
        watershed_seg = self.segment_region_based(img, 0.16, 0.19)

        # compute sample segmentations for all stored weights
        model_weights = os.listdir(self.path_model_weights)

        segs = [watershed_seg]
        for m in model_weights:
            model_pred = UNetv1(input_size=img_pad.shape[1:3] + (1,), inference=True)
            model_pred.load_weights(os.path.join(self.path_model_weights, m))
            y_pred = model_pred.predict(img_pad)
            seg = (self.undo_padding(y_pred) > 0.5).astype(int)
            segs.append(seg)

        # TODO: This could be done with tkinter buttons with images
        # display different segmentation methods (watershed + NN trained for different cell types)
        labels = ['watershed']
        labels += [mw.split('.')[0].split('_')[-1] for mw in model_weights]
        num_subplots = int(np.ceil(np.sqrt(len(segs))))
        plt.figure(figsize=(10, 10))
        for i, s in enumerate(segs):
            plt.subplot(num_subplots, num_subplots, i + 1)
            plt.imshow(img)
            plt.contour(s, [0.5], colors='r', linewidths=0.5)
            if i == 0:
                plt.title('watershed')
            else:
                plt.title('model trained for ' + labels[i])
            plt.xticks([])
            plt.yticks([])
        channel = os.path.basename(os.path.dirname(path_to_cutouts))
        plt.suptitle(f'Select model weights for channel: {channel}')
        rax = plt.axes([0.3, 0.01, 0.3, 0.08])
        # visibility = [False for i in range(len(segs))]
        check = RadioButtons(rax, labels)

        plt.show()

        # extract selected segmentation method from output of RadioButton
        if check.value_selected == 'watershed':
            self.model_weights = 'watershed'
        else:
            # extract the path
            ix_model_weights = np.where([check.value_selected == l for l in labels])[0][0]
            sel_model_weights = model_weights[ix_model_weights - 1]
            self.model_weights = os.path.join(self.path_model_weights, sel_model_weights)
