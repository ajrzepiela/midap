import os
import argparse
from pathlib import Path
from typing import Union

# to get all subclasses
from midap.tracking import *
from midap.tracking import base_tracking, cell_props
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
    connectivity = 1
    target_size = None
    input_size = None

    # Process
    tr = class_instance(imgs=img_names_sort, segs=seg_names_sort, model_weights=model_file, input_size=input_size,
                        target_size=target_size, connectivity=connectivity)
    data_file, csv_file = tr.track_all_frames(output_folder)

    # add the region props
    if data_file is not None and csv_file is not None:
        props = cell_props.CellProps(data_file=data_file, csv_file=csv_file)
        props.add_cell_probs(csv_file)


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
