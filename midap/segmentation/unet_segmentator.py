import io
import os
from typing import Collection, Union

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
from skimage.filters import sobel
from skimage.segmentation import watershed
from tqdm import tqdm

from .base_segmentator import SegmentationPredictor
from ..networks.unets import UNetv1


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

    def set_segmentation_method(self, path_to_cutouts: Union[str, bytes, os.PathLike]):
        """
        Performs the weight selection for the segmentation network. A custom method should use this function to set
        self.segmentation_method to a function that takes an input images and returns a segmentation of the image,
        i.e. an array in the same shape but with values only 0 (no cell) and 1 (cell)
        :param path_to_cutouts: The directory in which all the cutout images are
        """

        # check if we even need to select
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
            img_pad = self.pad_image(img)

            # Get all the labels
            labels = ['watershed']
            model_weights = os.listdir(self.path_model_weights)
            labels += [mw.split('.')[0].split('_')[-1] for mw in model_weights]

            # create the segmentations
            watershed_seg = self.segment_region_based(img, 0.16, 0.19)
            segs = [watershed_seg]
            for m in model_weights:
                model_pred = UNetv1(input_size=img_pad.shape[1:3] + (1,), inference=True)
                model_pred.load_weights(os.path.join(self.path_model_weights, m))
                y_pred = model_pred.predict(img_pad)
                seg = (self.undo_padding(y_pred) > 0.5).astype(int)
                segs.append(seg)

            # now we create a plot that can be used as a button image
            buffers = []
            for seg, model_name in zip(segs, labels):
                plt.figure(figsize=(3, 3))
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
                                              button_color=(
                                              sg.theme_background_color(), sg.theme_background_color()),
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

            # set the model weights
            if marked == 'watershed':
                self.model_weights = 'watershed'
            else:
                ix_model_weights = np.where([marked == l for l in labels])[0][0]
                sel_model_weights = model_weights[ix_model_weights - 1]
                self.model_weights = os.path.join(self.path_model_weights, sel_model_weights)

        # set the segmentation method
        if self.model_weights == 'watershed':
            self.segmentation_method = self.seg_method_watershed
        else:
            self.segmentation_method = self.seg_method_unet

    def seg_method_unet(self, imgs_in: Collection[np.ndarray]):
        """
        Performs image segmentation with unet and the selected model weights
        :param imgs_in: List of input images
        :return: List of segmentations
        """

        # pad the images
        imgs_pad = []
        for img in imgs_in:
            img = self.scale_pixel_vals(img)
            img_pad = self.pad_image(img)
            imgs_pad.append(img_pad)
        imgs_pad = np.concatenate(imgs_pad)

        # segments
        model_pred = UNetv1(input_size=imgs_pad.shape[1:3] + (1,), inference=True)
        model_pred.load_weights(self.model_weights)
        y_preds = model_pred.predict(imgs_pad, batch_size=1, verbose=1)

        # remove tha padding and transform to segmentation
        segs = []
        for i, y in enumerate(y_preds):
            seg = (self.undo_padding(y[None, ...]) > 0.5).astype(int)
            segs.append(seg)

        return segs

    def seg_method_watershed(self, imgs_in: Collection[np.ndarray]):
        """
        Performs watershed segmentation with scaling
        :param imgs_in: List of input images
        :return: List of segmentations
        """

        segs = []
        for img_in in tqdm(imgs_in):
            img = self.scale_pixel_vals(img_in)
            seg = self.segment_region_based(img, 0.16, 0.19)
            segs.append(seg)

        return segs

    def segment_region_based(self, img, min_val=40., max_val=50.):
        """
        Performs skimage's watershed segmentation on an image
        :param img: input image as an array
        :param min_val: minimum value used for the markers
        :param max_val: maximum value used for the markers
        :returns: the segmentation of the image
        """
        elevation_map = sobel(img)
        markers = np.zeros_like(img)
        markers[img < min_val] = 1
        markers[img > max_val] = 2
        segmentation = watershed(elevation_map, markers)
        return (segmentation <= 1).astype(int)

    def pad_image(self, img: np.ndarray):
        """
        Pad the image in mirror padding to the next higher number that is divisible by the set div attribute
        :param img: The input image as array
        :returns: The padded image
        """

        # get the new shape
        new_shape = (int(np.ceil(img.shape[0] / self.div) * self.div),
                     int(np.ceil(img.shape[1] / self.div) * self.div))

        # store values to remove padding later
        self.row_shape = img.shape[0]
        self.col_shape = img.shape[1]

        # get the padded image
        img_pad = np.pad(img, [[0, new_shape[0] - self.row_shape], [0, new_shape[1] - self.col_shape]], mode="reflect")

        # add batch and channel dim
        return img_pad[None,...,None]

    def undo_padding(self, img_pad: np.ndarray):
        """
        Reverses the padding added by <pad_image>
        :param img_pad: padded image as array
        :returns: The image without padding
        """
        img_unpad = img_pad[0, :self.row_shape, :self.col_shape, 0]
        return img_unpad
