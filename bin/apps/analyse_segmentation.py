from skimage import io
from skimage.measure import regionprops

import os
import numpy as np

from tqdm import tqdm

import pandas as pd
import argparse

import sys
from midap.utils import get_logger

# Functions
###########

def count_cells(seg):
    """
    Calculate the cell count of a segmentation
    :param seg: The input segmentation
    :returns: The number of cells found in the segmentation
    """
    return (len(np.unique(seg)) - 1)


def count_killed(seg):
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

# main
######

if __name__ == "__main__":

    # parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_seg", type=str, required=True, help="Path to the segmentation results.")
    parser.add_argument("--path_result", type=str, required=True, help="Path where the results should be stored")
    parser.add_argument("--loglevel", type=int, default=7, help="Loglevel of the script.")
    args = parser.parse_args()

    # logging
    logger = get_logger(__file__, args.loglevel)
    logger.info(f"Analysing segmentation of: {args.path_seg}")


    # computer number of living and killed cells
    num_cells = []
    num_killed = []

    # cycle through everything
    for p in tqdm(np.sort(os.listdir(args.path_seg))):
        num_cells.append(count_cells(io.imread(args.path_seg + p)))
        num_killed.append(count_killed(io.imread(args.path_seg + p)))

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
    df_cells.to_csv(os.path.join(args.path_result, 'cell_number.csv'))
