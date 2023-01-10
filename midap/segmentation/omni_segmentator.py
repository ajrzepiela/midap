import io
import os

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
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
            img = self.scale_pixel_vals(skio.imread(os.path.join(path_to_cutouts, path_img)))

            # display different segmentation models
            labels = ['bact_phase_cp', 'bact_fluor_cp', 'bact_phase_omni', 'bact_fluor_omni']
            buffers = []
            for model_name in labels:
                model = models.CellposeModel(gpu=False, model_type=model_name)
                # predict, we only need the mask, see omnipose tutorial for the rest of the args
                mask, _, _ = model.eval(img, channels=[0, 0], rescale=None, mask_threshold=-1,
                                        transparency=True, flow_threshold=0, omni=True, resample=True, verbose=0)
                # omni removes axes that are just 1
                seg = (mask > 0.5).astype(int)

                # now we create a plot that can be used as a button image
                plt.figure(figsize=(3,3))
                plt.imshow(img)
                plt.contour(seg, [0.5], colors='r', linewidths=0.5)
                plt.xticks([])
                plt.yticks([])
                plt.title(model_name)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                buffers.append(buf.read())

            # create the buttons
            num_cols = int(np.ceil(np.sqrt(len(labels))))
            buttons = []
            new_line = []
            for i, (b, l) in enumerate(zip(buffers, labels)):
                # the first button will be marked
                if i == 0:
                    new_line.append(sg.Button('', image_data=b,
                                              button_color=('black', 'yellow'),
                                              border_width=5, key=l))
                    marked = l
                else:
                    new_line.append(sg.Button('', image_data=b,
                                              button_color=(sg.theme_background_color(), sg.theme_background_color()),
                                              border_width=5, key=l))
                # create a new line if necessary
                if len(new_line) == num_cols:
                    buttons.append(new_line)
                    new_line = []
            # if the current line has element append
            if len(new_line) > 0:
                buttons.append(new_line)

            # The GUI
            layout = buttons
            layout += [[sg.Column([[sg.OK(), sg.Cancel()]], key="col_final")]]
            window = sg.Window('Segmentation Selection', layout, element_justification='c')
            print("Starting loop")
            # Event Loop
            while True:
                # Read event
                event, values = window.read()
                # break if we have one of these
                if event in (sg.WIN_CLOSED, 'Exit', 'Cancel', 'OK'):
                    break

                # get the last event
                for i, l in enumerate(labels):
                    # if the last event was an image button click, mark it
                    if event == l:
                        marked = l
                        break
                # maked button is highlighted
                for l in labels:
                    if marked == l:
                        window[l].update(button_color=('black', 'yellow'))
                    else:
                        window[l].update(button_color=(sg.theme_background_color(), sg.theme_background_color()))
            window.close()

            if event != 'OK':
                self.logger.critical("GUI was cancelled or unexpectedly closed, exiting...")
                exit(1)

            # set weights
            self.model_weights = marked

        # helper function for the seg method
        model = models.CellposeModel(gpu=False, model_type=self.model_weights)

        def seg_method(imgs):
            mask, _, _ = model.eval(imgs, channels=[0, 0], rescale=None, mask_threshold=-1,
                                    transparency=True, flow_threshold=0, omni=True, resample=True, verbose=0)

            # add the channel dimension and batch if it was 1
            return mask

        # set the segmentations method
        self.segmentation_method = seg_method
