import os

import numpy as np
import pandas as pd
import argparse

from skimage import io
from skimage.measure import regionprops
from pathlib import Path
from typing import Union
from tqdm import tqdm

from midap.utils import get_logger

# Functions
###########

def count_cells(seg: np.ndarray):
    """
    Calculate the cell count of a segmentation
    :param seg: The input segmentation
    :returns: The number of cells found in the segmentation
    """
    return len(np.unique(seg)) - 1


def count_killed(seg: np.ndarray):
    """
    Count the number of killed cells for a segmentation
    :param seg: The segmentation
    :returns: The number of kills
    """
    # compute regionprops
    regions = regionprops(seg)

    # compute ratio between minor and major axis 
    # (only of major axis length is larger than 0)
    minor_to_major = np.array(
        [r.minor_axis_length / r.major_axis_length for r in regions if r.major_axis_length > 0])

    # get number of cells with ratio > 0.5
    num_killed = len(np.where(minor_to_major > 0.7)[0])

    return num_killed

def main(path_seg: Union[str,bytes,os.PathLike], path_result: Union[str,bytes,os.PathLike], loglevel=7):
    """
    Analyses the segmentation images in a given folder
    :param path_seg: The directory containing the segmented images (labelled)
    :param path_result: The directory to save the results
    :param loglevel: The loglevel between 0 and 7 (defaults to 7)
    """

    # logging
    logger = get_logger(__file__, loglevel)
    logger.info(f"Analysing segmentation of: {path_seg}")

    # computer number of living and killed cells
    num_cells = []
    num_killed = []

    # transform to paths
    path_seg = Path(path_seg)
    path_result = Path(path_result)

    # cycle through everything
    for p in tqdm(sorted(path_seg.iterdir())):
        img = io.imread(p)
        num_cells.append(count_cells(img))
        num_killed.append(count_killed(img))

    # crete a dataframe
    num_cells = np.array(num_cells)
    num_killed = np.array(num_killed)
    num_living = num_cells - num_killed
    d = {
        'all cells': num_cells,
        'living cells': num_living,
        'killed cells': num_killed}
    df_cells = pd.DataFrame(data=d)

    # save
    df_cells.to_csv(path_result.joinpath("cell_number.csv"))

# main
######

if __name__ == "__main__":

    # parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_seg", type=str, required=True, help="Path to the segmentation results.")
    parser.add_argument("--path_result", type=str, required=True, help="Path where the results should be stored")
    parser.add_argument("--loglevel", type=int, default=7, help="Loglevel of the script.")
    args = parser.parse_args()

    # call the main with unpacked args
    main(**vars(args))


