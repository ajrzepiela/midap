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
from ipywidgets import interactive
from matplotlib.widgets import RadioButtons
from PIL import Image

from typing import Union, List


class SegmentationJupyter(object):
    """
    Loads text data from specific path
    """

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = path
        self.path_midap = '/Users/franziskaoschmann/Documents/midap'#'/cluster/project/sis/cdss/oschmanf/segmentation_training/midap'

        self.path_data_input = self.path + "/input_data/"
        self.path_data = self.path + "/raw_im/"
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

    def get_img_dims(self):
        """
        Extracts height and width of an image
        """
        img = Image.open(Path(self.path_data).joinpath(self.chosen_files[0]))
        self.img_height = img.height
        self.img_width = img.width
        # self.im_shape = np.array(np.array(im).shape)
        # if im.height != im.width:
        #     self.ix_height = np.where(self.im_shape == im.height)[0][0]
        #     self.ix_width = np.where(self.im_shape == im.width)[0][0]
        # elif im.height == im.width:
        #     self.ix_height = np.where(self.im_shape == im.height)[0][0]
        #     self.ix_width = np.where(self.im_shape == im.width)[0][1]

    def get_img_dims_ix(self, img):
        """
        Get indices of img width and height in img shape
        """
        self.img_shape = np.array(np.array(img).shape)
        if self.img_height != self.img_width:
            self.ix_height = np.where(self.img_shape == self.img_height)[0][0]
            self.ix_width = np.where(self.img_shape == self.img_width)[0][0]
        elif self.img_height == self.img_width:
            self.ix_height = np.where(self.img_shape == self.img_height)[0][0]
            self.ix_width = np.where(self.img_shape == self.img_width)[0][1]

    def get_ix_add_dims(self, img):
        """
        Get axis of additional dimensions (number of frames and number of channels)
        """
        ix_dims = np.arange(len(np.array(img).shape))
        self.ix_diff = list(set(ix_dims).difference(set([self.ix_height, self.ix_width])))
        #self.ix_diff = list(set(self.im_shape).difference(set([im.height, im.width])))

    def make_dropdown(self, size_dim, name_add_dims):
        drop_options = name_add_dims
        dropdown = widgets.Dropdown(
                        options=drop_options,
                        layout=widgets.Layout(width="50%"),
                        description="Dimension with length "+str(size_dim)+" is:",
                        style = {'description_width': '250px'},
                        )
        return dropdown

    def load_image_stack(self, ):
        ###Change name###

        # read image
        #self.im = Image.open(Path(self.path_data).joinpath(self.chosen_files[0]))
        self.img = io.imread(Path(self.path_data).joinpath(self.chosen_files[0]))
        self.get_img_dims()
        self.get_img_dims_ix(self.img)

        # get indices of additional dimensions
        self.get_ix_add_dims(self.img)

        # make dropdowns for number of channels and number of frames
        self.name_add_dims = ['num_channels', 'num_images']
        list_dropdowns = [self.make_dropdown(self.img_shape[ix_d], self.name_add_dims) for ix_d in self.ix_diff]
        self.hbox_dropdowns = widgets.HBox(list_dropdowns)

    def align_img_dims(self, img):
        # get indices of frame and channel indentifier
        name_dims = [c.value for c in self.hbox_dropdowns.children]

        # add length of axis to dict
        axis_length_dict = dict()
        for name, length in zip(name_dims, np.array(np.array(img).shape)[self.ix_diff]):
            axis_length_dict[name] = length

        # check which axes are currently present and create dict for assignment of new axes
        present_dims = list(set(self.name_add_dims)&(set(name_dims)))
        dims_assign_dict = dict()

        #for md, ixd in zip(present_dims, self.ix_diff):
        for md, ixd in zip(name_dims, self.ix_diff):
            dims_assign_dict[md] = ixd

        # move axes to (num_frames, height, width, num_channels)
        # ToDo: optimize and simplify
        img_clean = img.copy()
        img_clean = np.array(img_clean)

        if 'num_images' in dims_assign_dict.keys() and 'num_channels' in dims_assign_dict.keys():
            new_ax_im_len = np.where(np.array(self.img_shape) == axis_length_dict['num_images'])[0][0]
            img_clean = np.moveaxis(img_clean, new_ax_im_len, 0)
       
            new_ax_ch_len = np.where(np.array(img_clean.shape) == axis_length_dict['num_channels'])[0][0]
            img_clean = np.moveaxis(img_clean, new_ax_ch_len, -1)


        if 'num_images' in dims_assign_dict.keys() and 'num_channels' not in dims_assign_dict.keys():
            img_clean = np.moveaxis(img_clean, dims_assign_dict['num_images'], 0)

        if 'num_images' not in dims_assign_dict.keys() and 'num_channels' in dims_assign_dict.keys():
            img_clean = np.moveaxis(img_clean, dims_assign_dict['num_channels'], -1)

        # add axis in case one is missing
        if 'num_images' not in dims_assign_dict.keys():
            img_clean = np.expand_dims(img_clean, axis=0)

        if 'num_channels' not in dims_assign_dict.keys():
            img_clean = np.expand_dims(img_clean, axis=-1)

        return img_clean


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
        self.imgs_cut = np.array([
            img[self.y_min : self.y_max, self.x_min : self.x_max, 0] for img in imgs
        ])

    def show_all_cutouts(self):
        """
        Displays all cutouts with slider.
        """

        def f(i):
            return self.imgs_cut[int(i)]

        if len(self.imgs_cut) > 1:
            fig, ax = plt.subplots()
            controls = iplt.imshow(f, i=np.arange(0, len(self.imgs_cut) - 1))
        else:
            fig, ax = plt.subplots()
            plt.imshow(self.imgs_cut[0])


        plt.show()

    def save_cutouts(self):
        """
        Saves all cutouts to new folder.
        """

        # update path to cutout images and create dir if necessary
        self.path_cut = Path(self.path_cut).joinpath(self.chosen_files[0].split('.')[0])
        if not os.path.exists(self.path_cut): 
            os.makedirs(self.path_cut) 

        for i, cut in enumerate(self.imgs_cut):
            cut_scale = self.scale_pixel_val(cut)
            io.imsave(self.path_cut.joinpath("frame" + str('%(#)03d' % {'#': i}) + "_cut.png"), cut_scale)

    def get_seg_classes(self):
        segmentation_subclasses = [
            subclass
            for subclass in get_inheritors(base_segmentator.SegmentationPredictor)
        ]
        self.jupyter_seg_cls = [
            s.__name__
            for s in segmentation_subclasses
            if "Jupyter" in s.supported_setups
        ]

    def display_seg_classes(self):
        """
        Creates one checkbox per model type in folder and groups all checkboxes.
        """
        self.get_seg_classes()
        self.checkbox_widgets = [
            widgets.Checkbox(
                value=True, description=m, layout=widgets.Layout(width="100%")
            )
            for m in self.jupyter_seg_cls
        ]

        # Create observable output for the checkboxes
        self.checkbox_output_models = widgets.VBox(self.checkbox_widgets)

    def select_chosen_models(self):
        """
        Gets chosen file names for further analysis.
        """
        self.chosen_models = []
        for ch in self.checkbox_output_models.children:
            if ch.value:
                self.chosen_models.append(ch.description)

    # def display_seg_classes(self):
    #     self.get_seg_classes()
    #     self.out = widgets.Dropdown(
    #         options=self.jupyter_seg_cls,
    #         description="Segmentation method:",
    #         disabled=False,
    #     )

    def run_all_chosen_models(self):
        """
        Runs all pretrained models of chosen model types.
        """
        self.dict_all_models = {}
        for segmentation_class in self.chosen_models:
            segs = self.choose_segmentation_weights(segmentation_class)
            segs = dict(
                ("{}_{}".format(segmentation_class, k), v) for k, v in segs.items()
            )
            self.dict_all_models.update(segs)

    def compare_segmentations(self):
        """
        Displays two segmentations side-by-side for comparison of different pretrained models.
        """

        def f(a, b, c):
            fig = plt.figure(figsize=(12, 5))

            ax1 = fig.add_subplot(121)
            plt.imshow(self.dict_all_models[a][int(c)])
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.title(a)

            ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
            plt.imshow(self.dict_all_models[b][int(c)])
            plt.title(b)
            plt.show()

        self.output_seg_comp = interactive(
            f,
            a=widgets.Dropdown(
                options=self.dict_all_models.keys(),
                layout=widgets.Layout(width="50%"),
                description="Model 1",
            ),
            b=widgets.Dropdown(
                options=self.dict_all_models.keys(),
                layout=widgets.Layout(width="50%"),
                description="Model 2",
            ),
            c=widgets.IntSlider(
                min=0,
                max=(len(list(self.dict_all_models.values())[0]) - 1),
                description="Frame",
            ),
        )

    def choose_segmentation_weights(self, segmentation_class):
        # segmentation_class = self.out.label

        if segmentation_class == "OmniSegmentation":
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
        self.pred.set_segmentation_method_jupyter_all_imgs(self.path_cut)

        # self.segs = self.pred.segs

        return self.pred.segs

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
            options=list(self.dict_all_models.keys()),
            description="Model weights:",
            disabled=False,
            layout=widgets.Layout(width="100%"),
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
