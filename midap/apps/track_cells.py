import argparse
import numpy as np
import os
from pathlib import Path
from typing import Union

from skimage import io
from skimage.measure import label
from skimage.transform import resize

# to get all subclasses
from midap.tracking import *
from midap.tracking import base_tracking
from midap.utils import get_logger, get_inheritors


def main(path: Union[str, bytes, os.PathLike], tracking_class: str, loglevel=7):
    """
    The main function to run the tracking
    :param path: Path to the channel
    :param tracking_class: The name of the tracking class
    :param loglevel: The loglevel between 0 and 7, defaults to highest level
    """

    # logging
    logger = get_logger(__file__, loglevel)
    logger.info(f"Starting tracking for: {path}")

    # get the right subclass
    class_instance = None
    for subclass in get_inheritors(base_tracking.Tracking):
        if subclass.__name__ == tracking_class:
            class_instance = subclass

    # throw an error if we did not find anything
    if class_instance is None:
        raise ValueError(f"Chosen class does not exist: {tracking_class}")

    # Load data
    path = Path(path)
    images_folder = path.joinpath('cut_im')
    segmentation_folder = path.joinpath('seg_im')
    output_folder = path.joinpath('track_output')
    model_file = Path(__file__).absolute().parent.parent.parent.joinpath("model_weights",
                                                                         "model_weights_tracking",
                                                                         "unet_pads_track.hdf5")

    # glob all the cut images and segmented images
    img_names_sort = sorted(images_folder.glob('*frame*.png'))
    seg_names_sort = sorted(segmentation_folder.glob('*frame*.tif'))

    # Parameters:
    crop_size = (128, 128)
    connectivity = 1

    # Check if image resizing merges cells and adjust image size accordingly
    seg = io.imread(seg_names_sort[0])
    num_cells_orig = np.max(seg)
    num_cells_resize = np.max(label(resize(seg > 0, (512, 512))))

    num_cells_lower_thr = num_cells_orig * 0.99
    num_cells_upper_thr = num_cells_orig * 1.01

    if num_cells_resize == num_cells_orig:
        target_size = (512,512)
    else:
        img = io.imread(img_names_sort[0])
        row = int(img.shape[0]/8)*8
        col = int(img.shape[1]/8)*8
        target_size = (row, col)

    input_size = crop_size + (4,)

    # Process
    tr = class_instance(img_names_sort, seg_names_sort, model_file, input_size, target_size, crop_size, connectivity)
    tr.track_all_frames(output_folder)


if __name__ == "__main__":
    # arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='path to folder for one with specific channel')
    parser.add_argument("--tracking_class", type=str, required=True,
                        help="Name of the class used for the cell tracking. Must be defined in a file of "
                             "midap.tracking and a subclass of midap.tracking.Tracking")
    parser.add_argument("--loglevel", type=int, default=7, help="Loglevel of the script.")
    args = parser.parse_args()

    # call the main
    main(**vars(args))
