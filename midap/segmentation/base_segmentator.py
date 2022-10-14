import os
import re

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.widgets import RadioButtons
from skimage.filters import sobel
from skimage.measure import label, regionprops
from skimage.morphology import area_closing
from skimage.segmentation import watershed

from .model import unet_inference
from ..utils import get_logger

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = os.environ["__VERBOSE"]
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)


class SegmentationPredictor(object):
    """
    A class that performs the image segmentation of the cells
    """

    # this logger will be shared by all instances and subclasses
    logger = logger

    def __init__(self, path_model_weights, postprocessing, div=16, connectivity=1):
        """
        Initializes the SegmentationPredictor instance
        :param path_model_weights: Path to the model weights
        :param postprocessing: A flag for the postprocessing
        :param div: Divisor used for the padding of the images. Images will be padded to next higher number that is
                    divisible by div
        :param connectivity: The connectivity used for the segmentation, see skimage.measure.label
        """

        # set the params
        self.path_model_weights = path_model_weights
        self.postprocessing = postprocessing
        self.div = div
        self.connectivity = connectivity

        # This variable is used in case custom methods do not want the images padded (default)
        self.require_padding = False

        # params that will be set later
        self.model_weights = None
        self.segmentation_method = None

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
            model_pred = unet_inference(input_size=img_pad.shape[1:3] + (1,))
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

    def run_image_stack(self, channel_path):
        """
        Performs image segmentation, postprocessing and storage for all images found in channel_path
        :param channel_path: Directory of the channel used for the analysis
        """
        path_cut = os.path.join(channel_path, "cut_im")
        path_seg = os.path.join(channel_path, "seg_im")
        path_seg_track = os.path.join(channel_path, "input_ilastik_tracking")

        # get all the images to segment
        path_imgs = np.sort(os.listdir(path_cut))

        if self.model_weights is None or self.model_weights == 'watershed':
            self.logger.info('Image segmentation and storage')
            segs = []
            for p in path_imgs:
                # read the image
                img_in = io.imread(os.path.join(path_cut, p))

                # in case of custom method, we run it
                if self.model_weights is None:
                    seg = self.segmentation_method(img_in)
                # otherwise we use watershed
                else:
                    img = self.scale_pixel_vals(img_in)
                    seg = self.segment_region_based(img, 0.16, 0.19)

                # postprocessing
                if self.postprocessing:
                    seg = self.postprocess_seg(seg)

                # append and label
                segs.append(seg)
                seg_label = label(seg, connectivity=self.connectivity)

                # save the labeled segmentation
                io.imsave(os.path.join(path_seg, re.sub("(_cut.tif|_cut.png|.tif)", "_seg.png", p)),
                          seg_label.astype('uint16'), check_contrast=False)

            # save the full stack
            io.imsave(os.path.join(path_seg_track,
                                   re.sub("(_cut.tif|_cut.png|.tif)", "_full_stack_seg_bin.tiff", path_imgs[0])),
                      np.array(segs), check_contrast=False)

        # some unet weights were chosen
        else:
            self.logger.info('Image padding...')
            imgs_pad = []
            for p in path_imgs:
                img = self.scale_pixel_vals(io.imread(os.path.join(path_cut, p)))
                img_pad = self.pad_image(img)
                imgs_pad.append(img_pad)
            imgs_pad = np.concatenate(imgs_pad)

            self.logger.info('Image segmentation...')
            model_pred = unet_inference(input_size=imgs_pad.shape[1:3] + (1,))
            model_pred.load_weights(self.model_weights)
            y_preds = model_pred.predict(imgs_pad, batch_size=1, verbose=1)

            self.logger.info('Postprocessing and storage...')
            segs = []
            for i, y in enumerate(y_preds):
                # remove tha padding and transform to segmentation
                seg = (self.undo_padding(y[None,...]) > 0.5).astype(int)

                # postprocessing and labeling
                if self.postprocessing:
                    seg = self.postprocess_seg(seg)

                segs.append(seg)
                seg_label = label(seg, connectivity=self.connectivity)

                # save individual image
                io.imsave(os.path.join(path_seg, re.sub("(_cut.tif|_cut.png|.tif)", "_seg.png", path_imgs[i])),
                          seg_label.astype(np.uint8), check_contrast=False)

            # save the stacks
            io.imsave(os.path.join(path_seg_track,
                                   re.sub("(_cut.tif|_cut.png|.tif)", "_full_stack_cut.tiff", path_imgs[0])),
                      np.array(imgs_pad).astype(float), check_contrast=False)
            io.imsave(os.path.join(path_seg_track,
                                   re.sub("(_cut.tif|_cut.png|.tif)", "_full_stack_seg_bin.tiff", path_imgs[0])),
                      np.array(segs), check_contrast=False)
            io.imsave(os.path.join(path_seg_track,
                                   re.sub("(_cut.tif|_cut.png|.tif)", "_full_stack_seg_prob.tiff", path_imgs[0])),
                      y_preds.astype(float), check_contrast=False)

    def postprocess_seg(self, seg):
        """
        Performs postprocessing on a segmentation, e.g. remove segmentations that are too small and area closing
        :param seg: The input segmentation
        :returns: the processed segmentation
        """

        # remove small and big particels which are not cells
        label_objects = label(seg, connectivity=self.connectivity)
        sizes = np.bincount(label_objects.ravel())
        reg = regionprops(label_objects)
        areas = [r.area for r in reg]
        # min_size, max_size = np.quantile(areas, [0.01, 1.])
        min_size = np.quantile(areas, [0.01])
        # mask_sizes = (sizes > min_size)&(sizes < max_size)
        mask_sizes = (sizes > min_size)
        mask_sizes[0] = 0
        img_filt = (mask_sizes[label_objects] > 0).astype(int)

        # close small holes
        img_closed = area_closing(img_filt)
        return img_closed

    def scale_pixel_vals(self, img):
        """
        Scales the values of the pixels of an image such that they are between 0 and 1
        :param img: The input image as array
        :returns: The images with pixels scales between 0 and 1
        """
        img = np.array(img)
        return ((img - img.min()) / (img.max() - img.min()))

    def pad_image(self, img):
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

    def undo_padding(self, img_pad):
        """
        Reverses the padding added by <pad_image>
        :param img_pad: padded image as array
        :returns: The image without padding
        """
        img_unpad = img_pad[0, :self.row_shape, :self.col_shape, 0]
        return img_unpad
