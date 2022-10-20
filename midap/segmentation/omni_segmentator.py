import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.widgets import RadioButtons

from cellpose import models
from .base_segmentator import SegmentationPredictor


class OmniSegmentation(SegmentationPredictor):
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

        # display different segmentation methods (watershed + NN trained for different cell types)
        labels = ['bact_phase_cp', 'bact_fluor_cp', 'bact_phase_omni', 'bact_fluor_omni']
        segs = []
        for model_name in labels:
            model = models.CellposeModel(gpu=False, model_type=model_name)
            # predict, we only need the mask, see omnipose tutorial for the rest of the args
            mask, _, _ = model.eval(img, channels=[0, 0], rescale=None, mask_threshold=-1,
                                    transparency=True, flow_threshold=0, omni=True, resample=True, verbose=0)
            # omni removes axes that are just 1
            seg = (mask > 0.5).astype(int)
            segs.append(seg)

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

        # helper function for the seg method
        model = models.CellposeModel(gpu=False, model_type=check.value_selected)
        def seg_method(img):
            mask, _, _ = model.eval(img, channels=[0, 0], rescale=None, mask_threshold=-1,
                                    transparency=True, flow_threshold=0, omni=True, resample=True, verbose=0)

            # add the channel dimension and batch if it was 1
            return mask

        # set the segmentations method
        self.segmentation_method = seg_method
