import os
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
import numpy as np

from midap.utils import get_inheritors
from midap.segmentation import *
from midap.segmentation import base_segmentator
from midap.apps import segment_cells

import ipywidgets as widgets
from matplotlib.widgets import RadioButtons

from typing import Union, List


class SegmentationJupyter(object):
    """
    Loads text data from specific path
    """

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = path
        self.path_data = path + "/raw_im/"
        self.path_midap = "/Users/franziskaoschmann/Documents/midap/"

        self.path_cut = self.path + "/cut_im/"
        self.path_seg = self.path + "/seg_im/"
        os.makedirs(self.path_cut, exist_ok=True)
        os.makedirs(self.path_seg, exist_ok=True)

    def display_input_filenames(self):
        """
        Creates checkboxes for of all files in given folder.
        """
        self.get_input_files()
        self.create_checkboxes()

    def get_input_files(self):
        """
        Extracts input file names (except e.g. hidden files).
        """
        self.input_files = [
            f for f in os.listdir(self.path_data) if not f.startswith((".", "_"))
        ]

    def create_checkboxes(self):
        """
        Creates one checkbox per file in folder and groups all checkboxes.
        """
        self.checkbox_widgets = [
            widgets.Checkbox(
                value=True, description=f, layout=widgets.Layout(width="100%")
            )
            for f in self.input_files
        ]

        # Create observable output for the checkboxes
        self.checkbox_output = widgets.VBox(self.checkbox_widgets)

    def select_chosen_filenames(self):
        """
        Gets chosen file names for further analysis.
        """
        self.chosen_files = []
        for ch in self.checkbox_output.children:
            if ch.value:
                self.chosen_files.append(ch.description)
        self.chosen_files = np.sort(self.chosen_files)

    def load_all_images(self):
        """
        Load all chosen image files.
        """
        self.all_imgs = [
            io.imread(Path(self.path_data).joinpath(c)) for c in self.chosen_files
        ]

    def show_example_image(self, img: np.ndarray):
        """
        Displays example image.
        """
        _, self.ax = plt.subplots()
        self.ax.imshow(img)
        plt.show()

    def get_corners_cutout(self):
        """
        Gets axis limits for current zoom-in.
        """
        xlim = [int(xl) for xl in self.ax.get_xlim()]
        ylim = [int(yl) for yl in self.ax.get_ylim()]

        self.x_min = np.min(xlim)
        self.x_max = np.max(xlim)

        self.y_min = np.min(ylim)
        self.y_max = np.max(ylim)

    def make_cutouts(self, imgs: List[np.ndarray]):
        """
        Generates cutouts for all images.
        """
        self.imgs_cut = [
            img[self.y_min : self.y_max, self.x_min : self.x_max] for img in imgs
        ]

    def show_all_cutouts(self):
        """
        Displays all cutouts with slider.
        """

        def f(i):
            return self.imgs_cut[int(i)]

        fig, ax = plt.subplots()
        controls = iplt.imshow(f, i=np.arange(0, len(self.imgs_cut) - 1))

        plt.show()

    def save_cutouts(self):
        """
        Saves all cutouts to new folder.
        """
        for file, cut in zip(self.chosen_files, self.imgs_cut):
            cut_scale = self.scale_pixel_val(cut)
            io.imsave(self.path_cut + file.split(".")[0] + "_cut.png", cut_scale)

    def get_seg_classes(self):
        segmentation_subclasses = [
            subclass
            for subclass in get_inheritors(base_segmentator.SegmentationPredictor)
        ]
        self.family_seg_cls = [
            s.__name__
            for s in segmentation_subclasses
            if "Family_Machine" in s.supported_setups
        ]

    def display_seg_classes(self):
        self.get_seg_classes()
        self.out = widgets.Dropdown(
            options=self.family_seg_cls,
            description="Segmentation method:",
            disabled=False,
        )

    def choose_segmentation_weights(self):
        segmentation_class = self.out.label

        if segmentation_class == "HybridSegmentation":
            path_model_weights = Path(self.path_midap).joinpath(
                "model_weights", "model_weights_hybrid"
            )
        elif segmentation_class == "OmniSegmentation":
            path_model_weights = Path(self.path_midap).joinpath(
                "model_weights", "model_weights_omni"
            )
        else:
            path_model_weights = Path(self.path_midap).joinpath(
                "model_weights", "model_weights_legacy"
            )

        # define variables
        postprocessing = False
        network_name = None
        img_threshold = 255

        # get the right subclass
        class_instance = None
        for subclass in get_inheritors(base_segmentator.SegmentationPredictor):
            if subclass.__name__ == segmentation_class:
                class_instance = subclass

        # throw an error if we did not find anything
        if class_instance is None:
            raise ValueError(f"Chosen class does not exist: {segmentation_class}")

        # get the Predictor
        self.pred = class_instance(
            path_model_weights=path_model_weights,
            postprocessing=postprocessing,
            model_weights=network_name,
            img_threshold=img_threshold,
            jupyter=False,
        )

        # select the segmentor
        self.pred.set_segmentation_method_jupyter(self.path_cut)

        self.segs = self.pred.segs

    def display_segmentations(self):
        num_col = int(np.ceil(np.sqrt(len(self.segs))))
        plt.figure(figsize=(10, 10))
        for i, k in enumerate(self.segs.keys()):
            plt.subplot(num_col, num_col, i + 1)
            plt.imshow(self.segs[k])
            plt.title(k)
            plt.xticks([])
            plt.yticks([])

        plt.show()

    def display_buttons_weights(self):
        self.out_weights = widgets.RadioButtons(
            options=list(self.segs.keys()), description="Model weights:", disabled=False
        )

    def segment_all_images(self):
        self.pred.run_image_stack_jupyter(
            self.imgs_cut, self.out_weights.label, clean_border=False
        )

    def show_segmentations(self):
        def f(i):
            return self.pred.mask[int(i)]

        fig, ax = plt.subplots()
        controls = iplt.imshow(f, i=np.arange(0, len(self.pred.mask) - 1))

        plt.show()

    def scale_pixel_val(self, img):
        """
        Rescale the pixel values of the image
        :param img: The input image as array
        :returns: The images with pixels scales to standard RGB values
        """
        img_scaled = (255 * ((img - np.min(img)) / np.max(img - np.min(img)))).astype(
            "uint8"
        )
        return img_scaled
