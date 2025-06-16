import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from cellpose_omni import models
from typing import Union

import torch

from .base_segmentator import SegmentationPredictor
from ..utils import GUI_selector
import platform


class OmniSegmentation(SegmentationPredictor):
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

        if platform.processor() == "arm":
            self.gpu_available = torch.backends.mps.is_available()
        else:
            self.gpu_available = torch.cuda.is_available()

    def set_segmentation_method(self, path_to_cutouts):
        """
        Performs the weight selection for the segmentation network. A custom method should use this function to set
        self.segmentation_method to a function that takes an input images and returns a segmentation of the image,
        i.e. an array in the same shape but with values only 0 (no cell) and 1 (cell)
        :param path_to_cutouts: The directory in which all the cutout images are
        """

        if self.model_weights is None:
            self.logger.info("Selecting weights...")

            # get the image that is roughly in the middle of the stack
            list_files = np.sort(os.listdir(path_to_cutouts))
            # take the middle image (but round up, if there are only 2 we want the second)
            if len(list_files) == 1:
                ix_half = 0
            else:
                ix_half = int(np.ceil(len(list_files) / 2))

            path_img = list_files[ix_half]

            # scale the image and pad
            img = self.scale_pixel_vals(
                io.imread(os.path.join(path_to_cutouts, path_img))
            )

            # display different segmentation models
            label_dict = {
                #"bact_phase_cp": "bact_phase_cp",
                #"bact_fluor_cp": "bact_fluor_cp",
                "bact_phase_omni": "bact_phase_omni",
                "bact_fluor_omni": "bact_fluor_omni",
            }
            for custom_model in Path(self.path_model_weights).iterdir():
                    if custom_model.is_file() and custom_model.suffix == "" and not custom_model.name.startswith("."): # Filter out any other files that contain no weights (i.e .jsons)
                        label_dict.update({custom_model.name: custom_model})
            figures = []
            for model_name, model_path in label_dict.items():
                self.logger.info("Try model: " + str(model_name))
                model = self._build_cellpose_model(
                    model_path, gpu=self.gpu_available
                )

                # predict – adjust image if the model expects 2 channels
                test_img = (np.stack([img, img], -1)
                            if model.nchan == 2 and img.ndim == 2 else img)
                mask, _, _ = model.eval(
                    test_img,
                    channels=[0, 0],
                    rescale=None,
                    mask_threshold=-1,
                    transparency=True,
                    flow_threshold=0,
                    omni=True,
                    resample=True,
                    verbose=0,
                )
                # omni removes axes that are just 1
                seg = (mask > 0.5).astype(int)

                # now we create a plot that can be used as a button image
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(img)
                ax.contour(seg, [0.5], colors="r", linewidths=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(model_name)
                figures.append(fig)

            # Title for the GUI
            channel = os.path.basename(os.path.dirname(path_to_cutouts))
            # if we just got the chamber folder, we need to go one more up
            if channel.startswith("chamber"):
                channel = os.path.basename(
                    os.path.dirname(os.path.dirname(path_to_cutouts))
                )
            title = f"Segmentation Selection for channel: {channel}"

            # start the gui
            marked = GUI_selector(
                figures=figures, labels=list(label_dict.keys()), title=title
            )

            # set weights
            self.model_weights = label_dict[marked]

        # helper function for the seg method
        model = self._build_cellpose_model(
            self.model_weights, gpu=self.gpu_available
        )

        def seg_method(imgs):
            # scale all the images
            imgs = [self.scale_pixel_vals(img) for img in imgs]
            if model.nchan == 2 and imgs[0].ndim == 2:        # duplicate chan
                imgs = [np.stack([im, im], axis=-1) for im in imgs]

            # we catch here ValueErrors because omni can fail at masking when there are no cells
            try:
                mask, _, _ = model.eval(
                    imgs,
                    channels=[0, 0],
                    rescale=None,
                    mask_threshold=-1,
                    transparency=True,
                    flow_threshold=0,
                    omni=True,
                    resample=True,
                    verbose=0,
                )
            except ValueError:
                self.logger.warning("Segmentation failed, returning empty mask!")
                mask = np.zeros((len(imgs),) + imgs[0].shape, dtype=int)

            # add the channel dimension and batch if it was 1
            return mask

        # set the segmentations method
        self.segmentation_method = seg_method

    # --------------------------------------------------------------
    # helper: build Cellpose / Omnipose model with correct nchan
    # --------------------------------------------------------------
    def _build_cellpose_model(
        self, model_id: Union[str, os.PathLike], *, gpu: bool
    ):
        """
        Create a CellposeModel with nchan adjusted to the checkpoint.
        Falls back to 1-channel if the weight file cannot be inspected.
        """
        built_in_two_chan = {
            "cyto", "cyto2", "cytotorch_0",
            "bact_phase_cp", "bact_fluor_cp",
        }

        model_id = str(model_id)
        nchan = 1

        if Path(model_id).is_file():                      # custom file
            try:
                ckpt = torch.load(model_id, map_location="cpu")
                first_conv = next(v for v in ckpt["state_dict"].values()
                                   if v.ndim == 4)
                nchan = first_conv.shape[1]
            except Exception:
                pass
            return models.CellposeModel(
                gpu=gpu, nchan=nchan, pretrained_model=model_id
            )

        if model_id in built_in_two_chan:                 # built-in name
            nchan = 2

        return models.CellposeModel(gpu=gpu, nchan=nchan, model_type=model_id)
