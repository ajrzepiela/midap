import os
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from tqdm import tqdm

from ..utils import get_logger

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = int(os.environ["__VERBOSE"])
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)

class CellProps:
    """
    A class that load the label stack and csv files and adds the cell properties
    """

    # this logger will be shared by all instances and subclasses
    logger = logger

    def __init__(self, data_file: Union[str, bytes, os.PathLike], csv_file: Union[str, bytes, os.PathLike]):
        """
        Inits the class with the file names, checks the existence
        :param data_file: The path to the h5 data file
        :param csv_file: The path to the csv table
        """

        self.data_file = Path(data_file)
        self.csv_file = Path(csv_file)

        if not self.data_file.is_file():
            raise FileNotFoundError(f"Data file does not exist: {self.data_file}")
        if not self.csv_file.is_file():
            raise FileNotFoundError(f"CSV file does not exist: {self.csv_file}")

    def add_cell_probs(self, out_file: Union[str, bytes, os.PathLike, None] = None):
        """
        Adds the cellprobs to the table
        :param out_file: The CSV output file, if None, the original CSV file is overwritten
        """

        # read the data
        df = pd.read_csv(self.csv_file)

        # init the new columns
        new_cols = ['area', 'edges_min_row', 'edges_min_col', 'edges_max_row', 'edges_max_col', 'intensity_max',
                    'intensity_mean', 'intensity_min', 'minor_axis_length', 'major_axis_length']
        for new_col in new_cols:
            df[new_col] = np.nan

        # go through cells
        with h5py.File(self.data_file, "r") as f:
            labels = f["labels"]
            images = f["images"]

            self.logger.info("Calculating cell properties...")
            for frame_num, (l, i) in tqdm(enumerate(zip(labels, images)), total=len(labels)):
                cell_props = regionprops(l, intensity_image=i)
                for prop in cell_props:
                    # select current cell and check
                    row_selector = (df["frame"] == frame_num) & (df["trackID"] == prop.label)
                    assert np.sum(row_selector) == 1

                    # set all attributes
                    df.loc[row_selector, 'x'] = prop.centroid[0]
                    df.loc[row_selector, 'y'] = prop.centroid[1]
                    df.loc[row_selector, 'edges_min_row'] = prop.bbox[0]
                    df.loc[row_selector, 'area'] = prop.area
                    df.loc[row_selector, 'edges_min_col'] = prop.bbox[1]
                    df.loc[row_selector, 'edges_max_row'] = prop.bbox[2]
                    df.loc[row_selector, 'edges_max_col'] = prop.bbox[3]
                    df.loc[row_selector, 'intensity_max'] = prop.intensity_max
                    df.loc[row_selector, 'intensity_mean'] = prop.intensity_mean
                    df.loc[row_selector, 'intensity_min'] = prop.intensity_min
                    df.loc[row_selector, 'minor_axis_length'] = prop.minor_axis_length
                    df.loc[row_selector, 'major_axis_length'] = prop.major_axis_length

        # check if all properties are set
        for col in new_cols:
            if np.any(np.isnan(df[col].values)):
                logger.warning(f"Not all values set in col: {col}")

        if out_file is None:
            df.to_csv(self.csv_file, index=True)
        else:
            df.to_csv(out_file, index=True)
