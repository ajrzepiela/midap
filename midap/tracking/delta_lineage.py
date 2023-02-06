import os
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label

from ..utils import get_logger

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = int(os.environ["__VERBOSE"])
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)


class DeltaTypeLineages:
    """
    A class to generate lineages based on trackinng outputs.
    """

    # this logger will be shared by all instances and subclasses
    logger = logger

    def __init__(self, inputs: np.ndarray, results: np.ndarray, generate_lineage=True):
        """
        Initializes the class
        :param inputs: input array for tracking network
        :param results: output array of tracking network
        :param generate_lineage: Generate the lineages immediately, defaults to True
        """

        self.results = results

        # we append a input for the last frame
        last_frame = np.zeros_like(inputs[:1])
        last_frame[0,...,0] = inputs[-1,...,2]
        last_frame[0,...,1] = label(inputs[-1,...,3])

        self.inputs = np.concatenate([inputs, last_frame], axis=0)
        self.n_frames = len(self.inputs)

        # in this label stack all cells with the same ID are the same cell
        self.label_stack = np.zeros(self.inputs.shape[:-1])

        # get the dataframe
        self.track_output = self.init_dataframe()

        # generate the lineages
        if generate_lineage:
            self.generate_lineages()

    def init_dataframe(self):
        """
        Initialize dataframe for tracking output.
        :return: An empty dataframe with column labels
        """

        columns = ['frame', 'labelID', 'trackID', 'lineageID', 'trackID_d1', 'trackID_d2', 'split',
                   'trackID_mother', 'area', 'edges_min_row', 'edges_min_col', 'edges_max_row',
                   'edges_max_col', 'intensity_max', 'intensity_mean', 'intensity_min',
                   'minor_axis_length', 'major_axis_length',
                   'first_frame', 'last_frame']
        return pd.DataFrame(columns=columns)

    def generate_lineages(self):
        """
        Generates lineages based on output of tracking (U-Net) network.
        """

        # init the global unique ID and the track ID that track the cell through multiple cells
        global_id = 1
        track_id = 1

        # this goes through all labeled input the last
        for frame_num, label_inp in enumerate(self.inputs[...,1]):
            # we get all cells in the frame, first element is background
            current_local_ids = np.unique(label_inp)[1:]

            # cycle through all local ids
            for local_id in current_local_ids:
                # track the cell if it's not already part of a lineage
                if local_id not in self.track_output.loc[self.track_output["frame"] == frame_num, "labelID"].values:
                    global_id, track_id = self._track_cell(frame_index=frame_num, cell_label=local_id,
                                                           global_id=global_id, track_id=track_id)

    def _track_cell(self, frame_index: int, cell_label: int, global_id: int, track_id: int,
                    first_frame: Optional[int]=None, lineage_id: Optional[int]=None, mother_id: Optional[int]=None):
        """
        Tracks a cell through the results recursively
        :param frame_index: The index of the frame where the cell is located
        :param cell_label: The label of the cell in the frame given by frame_index
        :param global_id: The global ID for this cell
        :param track_id: The tracking ID for this cell
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

        # we get the cell properties
        cell = (self.inputs[frame_index,...,1] == cell_label)
        cell_props = regionprops(cell.astype(int), intensity_image=self.inputs[frame_index, :, :, 0])[0]

        # update label stack
        self.label_stack[frame_index, cell] = track_id

        # add cell to output
        self.track_output.loc[global_id, 'frame'] = frame_index
        self.track_output.loc[global_id, 'labelID'] = cell_label
        self.track_output.loc[global_id, 'trackID'] = track_id
        self.track_output.loc[global_id, 'lineageID'] = lineage_id
        self.track_output.loc[global_id, 'area'] = cell_props.area
        self.track_output.loc[global_id, 'edges_min_row'] = cell_props.bbox[0]
        self.track_output.loc[global_id, 'edges_min_col'] = cell_props.bbox[1]
        self.track_output.loc[global_id, 'edges_max_row'] = cell_props.bbox[2]
        self.track_output.loc[global_id, 'edges_max_col'] = cell_props.bbox[3]
        self.track_output.loc[global_id, 'intensity_max'] = cell_props.intensity_max
        self.track_output.loc[global_id, 'intensity_mean'] = cell_props.intensity_mean
        self.track_output.loc[global_id, 'intensity_min'] = cell_props.intensity_min
        self.track_output.loc[global_id, 'minor_axis_length'] = cell_props.minor_axis_length
        self.track_output.loc[global_id, 'major_axis_length'] = cell_props.major_axis_length
        self.track_output.loc[global_id, 'first_frame'] = first_frame
        if mother_id is not None:
            self.track_output.loc[global_id, 'trackID_mother'] = mother_id

        # last frame
        if frame_index == self.n_frames - 1:
            # no split
            self.track_output.loc[global_id, 'split'] = 0
            # update the last frame for all previous cells
            self.track_output.loc[self.track_output['trackID'] == track_id, 'last_frame'] = frame_index

            # return new global id and track id
            return global_id + 1, track_id + 1

        # Generate binary masks for mother and daughter cells
        daughter_1 = self.results[frame_index,:, :, 0] == cell_label
        daughter_2 = self.results[frame_index,:, :, 1] == cell_label

        # Case 1: only daughter 1 is present
        if daughter_1.sum() > 0 and daughter_2.sum() == 0:
            # no split occured
            self.track_output.loc[global_id, 'split'] = 0
            # get the local ID in the next frame
            new_local_id = self.get_id_from_mask(label_img=self.inputs[frame_index + 1, ..., 1], mask=daughter_1)
            global_id, track_id = self._track_cell(frame_index=frame_index + 1, cell_label=new_local_id,
                                                   global_id=global_id + 1, first_frame=first_frame, track_id=track_id,
                                                   lineage_id=lineage_id)

        # Case 2: only daughter 2 is present
        elif daughter_1.sum() == 0 and daughter_2.sum() > 0:
            # no split occured
            self.track_output.loc[global_id, 'split'] = 0
            # get the local ID in the next frame
            new_local_id = self.get_id_from_mask(label_img=self.inputs[frame_index + 1, ..., 1], mask=daughter_2)
            global_id, track_id = self._track_cell(frame_index=frame_index + 1, cell_label=new_local_id,
                                                   global_id=global_id + 1, first_frame=first_frame, track_id=track_id,
                                                   lineage_id=lineage_id)

        # Case 3: cell split: both daughters are present
        elif daughter_1.sum() > 0 and daughter_2.sum() > 0:
            # split occured
            self.track_output.loc[global_id, 'split'] = 1
            # update the last frame for all previous cells
            self.track_output.loc[self.track_output['trackID'] == track_id, 'last_frame'] = frame_index

            # mother id for both cells
            mother_id = track_id

            # deal with daughter 1, get new local ID, set trackID of daughter for previous cells, tracl
            new_local_id = self.get_id_from_mask(label_img=self.inputs[frame_index + 1, ..., 1], mask=daughter_1)
            self.track_output.loc[self.track_output['trackID'] == mother_id, 'trackID_d1'] = track_id + 1
            global_id, track_id = self._track_cell(frame_index=frame_index + 1, cell_label=new_local_id,
                                                   global_id=global_id + 1, track_id=track_id + 1, lineage_id=lineage_id,
                                                   mother_id=mother_id)

            # deal with daughter 2, get new local ID, set trackID of daughter for previous cells, track
            new_local_id = self.get_id_from_mask(label_img=self.inputs[frame_index + 1, ..., 1], mask=daughter_2)
            # we do not need to increment track and global id here, since the previous call did that
            self.track_output.loc[self.track_output['trackID'] == mother_id, 'trackID_d2'] = track_id
            global_id, track_id = self._track_cell(frame_index=frame_index + 1, cell_label=new_local_id,
                                                   global_id=global_id, track_id=track_id, lineage_id=lineage_id,
                                                   mother_id=mother_id)

        # case 4: cell disappears
        elif daughter_1.sum() == 0 and daughter_2.sum() == 0:
            # no split occured
            self.track_output.loc[global_id, 'split'] = 0
            # update the last frame for all previous cells
            self.track_output.loc[self.track_output['trackID'] == track_id, 'last_frame'] = frame_index
            # update global and track id
            global_id += 1
            track_id += 1

        return global_id, track_id

    def get_id_from_mask(self, label_img, mask):
        """
        Given an labeled image and a mask, return the label of occuring in the mask,
        :param label_img: The labeled image
        :param mask: The mask to apply
        :return: The unique label of the mask, raises an error if multiple labels are within the mask
        """

        masked_id = np.unique(label_img[mask])

        assert len(masked_id) == 1

        return masked_id[0]

    def store_lineages(self, output_folder: Union[str, bytes, os.PathLike]):
        """
        Store tracking output files: labeled stack, tracking output, input files.
        :output_folder: Folder where to store the data
        """

        # transform to path
        output_folder = Path(output_folder)

        # save everything
        self.track_output.to_csv(output_folder.joinpath('track_output_delta.csv'), index=True)

        raw_inputs = self.inputs[:, :, :, 0]
        with h5py.File(output_folder.joinpath('raw_inputs_delta.h5'), 'w') as hf:
            hf.create_dataset('raw_inputs', data=raw_inputs)

        segs = self.inputs[0, :, :, 3]
        with h5py.File(output_folder.joinpath('segmentations_delta.h5'), 'w') as hf:
            hf.create_dataset('segmentations', data=segs)

        with h5py.File(output_folder.joinpath('label_stack_delta.h5'), 'w') as hf:
            hf.create_dataset('label_stack', data=self.label_stack)
