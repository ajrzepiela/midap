import os
from collections import defaultdict
from pathlib import Path
from shutil import rmtree
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
import tqdm
from skimage import io

from .bayesian_tracking import label_transform
from ..utils import get_logger

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = int(os.environ["__VERBOSE"])
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)

class STrackLineage(object):
    """
    This class transforms the strack output into the midap lineage format
    """

    # this logger will be shared by all instances and subclasses
    logger = logger

    def __init__(self, base_dir: Union[str, bytes, os.PathLike], imgs: np.ndarray, segs: np.ndarray,
                 remove_strack_output=True):
        """
        Inits the class
        :param base_dir: Path to the base directory of the channel
        :param imgs: Array of the original images, sorted
        :param segs: Array of the segmentations, sorted
        :param remove_strack_output: If True, the strack output will be removed
        """

        # set the attributes
        self.base_dir = Path(base_dir)
        self.remove_strack_output = remove_strack_output
        self.original_images = imgs
        self.segmented_images = segs

        # get some basic info
        self.n_frames = len(list(self.base_dir.joinpath("STrack").glob("tracking_table_*.csv"))) + 1

        # init the dataframe
        columns = ['frame', 'labelID', 'trackID', 'lineageID', 'trackID_d1', 'trackID_d2', 'split',
                   'trackID_mother', 'first_frame', 'last_frame']
        self.track_df = pd.DataFrame(columns=columns)

    def get_df_at_frame(self, frame: int):
        """
        opens the table file for a given frame
        :param frame: The frame number
        :return: A pandas dataframe
        """

        return pd.read_csv(self.base_dir.joinpath("STrack", f"tracking_table_time{frame}.csv"),
                           usecols=["Mask_nb", "Mother_mask"], dtype=int)

    def create_lineage_dicts(self):
        """
        For each frame, except the last, a dictionary is created that maps the old label id to the new label id.
        The new label is a list that is either empty, contains the new label id or the labels of the new cells that
        were created by splitting.
        :return: A list of dicts
        """

        # init the list
        lineage_dicts = []

        # loop over all frames
        for frame in range(1, self.n_frames):
            # get the dataframe
            df = self.get_df_at_frame(frame)
            lineage_dict = defaultdict(list)

            # loop over all cells
            for _, row in df.iterrows():
                mother_id = row["Mother_mask"]
                cell_id = row["Mask_nb"]
                lineage_dict[mother_id].append(cell_id)

            # add the dict to the list
            lineage_dicts.append(lineage_dict)

        return lineage_dicts

    def generate_midap_output(self):
        """
        Generate label stack based on tracking output.
        :return: The path to the generated csv and h5 file
        """

        self.logger.info('Generate lineages...')

        # create the lineage dicts
        lineage_dicts = self.create_lineage_dicts()

        # init all the ids
        global_id = 1
        track_id = 1

        # loop over all segmented images
        for frame_num, seg_im in tqdm.tqdm(enumerate(self.segmented_images)):
            current_local_ids = np.unique(seg_im)[1:]

            # track all cells
            for local_id in current_local_ids:
                # track the cell if it's not already part of a lineage
                if local_id not in self.track_df.loc[self.track_df["frame"] == frame_num, "labelID"].values:
                    global_id, track_id = self._track_cell(frame_index=frame_num, cell_label=local_id,
                                                           global_id=global_id, track_id=track_id,
                                                           lineage_dicts=lineage_dicts)

        # create the label stack
        label_stack = np.stack(self.segmented_images).astype(np.int32)
        label_transformations = []
        for _, row in self.track_df.iterrows():
            label_transformations.append((row["frame"], row["labelID"], row["trackID"]))
        label_transformations = np.array(label_transformations).astype(np.int32)
        if label_transformations.size > 0:
            label_transform(label_stack, label_transformations)

        # save the output
        data_file, csv_file = self.store_lineages(output_folder=self.base_dir, df=self.track_df, label_stack=label_stack,
                                                  raw_imgs=self.original_images,
                                                  segmentations=self.segmented_images)

        # remove the strack output if necessary
        if self.remove_strack_output:
            rmtree(self.base_dir.joinpath("STrack"), ignore_errors=True)

        return data_file, csv_file

    def _track_cell(self, frame_index: int, cell_label: int, global_id: int, track_id: int, lineage_dicts: list,
                    first_frame: Optional[int]=None, lineage_id: Optional[int]=None, mother_id: Optional[int]=None):
        """
        Tracks a cell through the results recursively
        :param frame_index: The index of the frame where the cell is located
        :param cell_label: The label of the cell in the frame given by frame_index
        :param global_id: The global ID for this cell
        :param track_id: The tracking ID for this cell
        :param lineage_dicts: A list of dicts that map the old label id to the new label id
        :param first_frame: The frame index of the first frame the cell appeared, defaults to frame_index
        :param lineage_id: The lineage ID of the current lineage
        :param mother_id: The optional tracking ID of the mother cell if the cell resulted from a split
        :return: The next unique global and tracking id
        """

        # set the lineage ID if necessary
        if lineage_id is None:
            lineage_id = track_id
        # get the frist frame if necessary
        if first_frame is None:
            first_frame = frame_index

        # add cell to output
        self.track_df.loc[global_id, 'frame'] = frame_index
        self.track_df.loc[global_id, 'labelID'] = cell_label
        self.track_df.loc[global_id, 'trackID'] = track_id
        self.track_df.loc[global_id, 'lineageID'] = lineage_id
        self.track_df.loc[global_id, 'first_frame'] = first_frame
        if mother_id is not None:
            self.track_df.loc[global_id, 'trackID_mother'] = mother_id

        # last frame
        if frame_index == self.n_frames - 1:
            # no split
            self.track_df.loc[global_id, 'split'] = 0
            # update the last frame for all previous cells
            self.track_df.loc[self.track_df['trackID'] == track_id, 'last_frame'] = frame_index

            # return new global id and track id
            return global_id + 1, track_id + 1

        # Case 1: only daughter 1 is present
        if len(lineage_dicts[frame_index][cell_label]) == 1:
            # no split occured
            self.track_df.loc[global_id, 'split'] = 0
            # get the local ID in the next frame
            new_local_id = lineage_dicts[frame_index][cell_label][0]
            global_id, track_id = self._track_cell(frame_index=frame_index + 1, cell_label=new_local_id,
                                                   global_id=global_id + 1, first_frame=first_frame, track_id=track_id,
                                                   lineage_id=lineage_id, lineage_dicts=lineage_dicts)

        # Case 2: cell split: both daughters are present
        elif len(lineage_dicts[frame_index][cell_label]) == 2:
            # split occured
            self.track_df.loc[global_id, 'split'] = 1
            # update the last frame for all previous cells
            self.track_df.loc[self.track_df['trackID'] == track_id, 'last_frame'] = frame_index

            # mother id for both cells
            mother_id = track_id

            # deal with daughter 1, get new local ID, set trackID of daughter for previous cells, tracl
            new_local_id = lineage_dicts[frame_index][cell_label][0]
            self.track_df.loc[self.track_df['trackID'] == mother_id, 'trackID_d1'] = track_id + 1
            global_id, track_id = self._track_cell(frame_index=frame_index + 1, cell_label=new_local_id,
                                                   global_id=global_id + 1, track_id=track_id + 1, lineage_id=lineage_id,
                                                   mother_id=mother_id, lineage_dicts=lineage_dicts)

            # deal with daughter 2, get new local ID, set trackID of daughter for previous cells, track
            new_local_id = lineage_dicts[frame_index][cell_label][1]
            # we do not need to increment track and global id here, since the previous call did that
            self.track_df.loc[self.track_df['trackID'] == mother_id, 'trackID_d2'] = track_id
            global_id, track_id = self._track_cell(frame_index=frame_index + 1, cell_label=new_local_id,
                                                   global_id=global_id, track_id=track_id, lineage_id=lineage_id,
                                                   mother_id=mother_id, lineage_dicts=lineage_dicts)

        # case 3: cell disappears
        elif len(lineage_dicts[frame_index][cell_label]) == 0:
            # no split occured
            self.track_df.loc[global_id, 'split'] = 0
            # update the last frame for all previous cells
            self.track_df.loc[self.track_df['trackID'] == track_id, 'last_frame'] = frame_index
            # update global and track id
            global_id += 1
            track_id += 1

        return global_id, track_id

    def store_lineages(self, output_folder: Union[str, bytes, os.PathLike], df: pd.DataFrame, label_stack: np.ndarray,
                       segmentations: np.ndarray, raw_imgs: np.ndarray):
        """
        Store tracking output files: labeled stack, tracking output, input files.
        :param output_folder: Folder where to store the data
        :param df: The pandas data frame to store as csv
        :param label_stack: The labelstack array to store
        :param segmentations: The segmentation array to store
        :param raw_imgs: The raw image array to store
        :return: The data file name and csv file name
        """

        # transform to path
        output_folder = Path(output_folder)

        # save everything
        csv_file = output_folder.joinpath("track_output_strack.csv")
        df.to_csv(csv_file, index=True, index_label="globalID")

        data_file = output_folder.joinpath("tracking_strack.h5")
        with h5py.File(data_file, "w") as hf:
            hf.create_dataset("images", data=raw_imgs.astype(np.float32), dtype=np.float32)
            hf.create_dataset("labels", data=label_stack.astype(np.int32), dtype=np.int32)

        with h5py.File(output_folder.joinpath("segmentations_strack.h5"), "w") as hf:
            hf.create_dataset("segmentations", data=segmentations.astype(np.int32), dtype=np.int32)

        return data_file, csv_file
