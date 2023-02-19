import os
from pathlib import Path
from shutil import copyfile
from typing import Union, Optional

import dask.array as da
import h5py
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info
from numba import njit

# Constants
###########

CORRECTION_SUFFIX = ".midap"


# Classes
#########

class TrackingData(object):
    """
    This class is designed to transform the MIDAP CSV into Napari compatible tracking data and back.
    It is also used to update linearges etc.
    """

    def __init__(self, csv_file: Union[str, bytes, os.PathLike]):
        """
        Inits the tracking data, if a already corrected file exists, it will read this
        :param csv_file: The original csv file
        """

        # read the file or the corrected file if it exits
        csv_file = Path(csv_file)
        if not csv_file.exists():
            raise FileNotFoundError(f"The CSV file does not exist: {csv_file}")

        # get the corrected file name
        if csv_file.suffix.endswith(CORRECTION_SUFFIX):
            self.corrected_file = csv_file
        else:
            self.corrected_file = csv_file.with_suffix(csv_file.suffix + CORRECTION_SUFFIX)

        # we read the corrected file if it exists
        if self.corrected_file.exists():
            self.track_df = pd.read_csv(self.corrected_file)
        else:
            self.track_df = pd.read_csv(csv_file)
            # save to corrected file already
            self.track_df.to_csv(self.corrected_file)

        # this is a list of mappings of track IDs old -> new to transform the label images on the fly
        self.transformation_file = self.corrected_file.parent.joinpath(".transformations.npy")
        self.track_id_transforms = []

        # counters to keep track of things
        self.next_lineage_id = self.track_df["lineageID"].max() + 1
        self.next_track_id = self.track_df["trackID"].max() + 1

    def get_number_of_cells(self, frame_number: int):
        """
        Returns the number of cells in a frame
        :param frame_number: The number of the frame
        :return: The number of cells in the frame
        """

        return np.sum(self.track_df["frame"] == frame_number)

    def get_number_of_orphans(self, frame_number: int, return_ids=False):
        """
        Calculates the number of orphans in a given frame. A orphan is a cell that has not parent, i.e. a cell who
        does not have a matching lineage ID in the previous frame
        :param frame_number: The number of the frame
        :param return_ids: If True, return the tracking IDs of the orphans
        :return: The number of orphans in the frame, if return_ids is True a list of tracking IDs is returned
        """

        current_frame = self.track_df[(self.track_df["first_frame"] == frame_number) &
                                      (self.track_df["frame"] == frame_number)]
        previous_frame = self.track_df[self.track_df["frame"] == frame_number - 1]

        # remove the cells exist in the previous frame
        orphans = []
        previous_lineage_ids = set(previous_frame["lineageID"].values)
        for current_track_id, current_lineage_id in zip(current_frame["trackID"].values,
                                                        current_frame["lineageID"].values):
            if current_lineage_id not in previous_lineage_ids:
                orphans.append(current_track_id)
        if return_ids:
            return orphans
        else:
            return len(orphans)

    def get_number_of_dying(self, frame_number: int, return_ids=False):
        """
        Calculates the number of dying cells in a frame, i.e. cells whose lineage ID does not continue in the next frame
        :param frame_number: The number of the frame
        :param return_ids: If True, return the tracking IDs of the orphans
        :return: The number of dying cells, if return_ids is True, a list of tracking ID is returned
        """

        current_frame = self.track_df[(self.track_df["last_frame"] == frame_number) &
                                      (self.track_df["frame"] == frame_number)]
        next_frame = self.track_df[self.track_df["frame"] == frame_number + 1]

        # remove the cells exist in the previous frame
        dying = []
        next_lineage_ids = set(next_frame["lineageID"].values)
        for current_track_id, current_lineage_id in zip(current_frame["trackID"].values,
                                                        current_frame["lineageID"].values):
            if current_lineage_id not in next_lineage_ids:
                dying.append(current_track_id)
        if return_ids:
            return dying
        else:
            return len(dying)

    def cell_in_frame(self, track_id: int, frame_number: int):
        """
        Checks if a track id is in a given frame
        :param track_id: The track id to check
        :param frame_number: The frame number to check
        :return: True if in frame otherwise False
        """

        return track_id in self.track_df[self.track_df["frame"] == frame_number]["trackID"].values

    def get_first_occurrence(self, track_id: int):
        """
        Returns the first occurrence of a cell
        :param track_id: The track id of the cell
        :return: The frame number of the first occurrence or None if the track ID does not exist
        """

        if track_id in self.track_df["trackID"].values:
            return int(self.track_df.iloc[(self.track_df["trackID"] == track_id).argmax()]["first_frame"])
        else:
            return None

    def get_last_occurrence(self, track_id: int):
        """
        Returns the last occurrence of a cell
        :param track_id: The track id of the cell
        :return: The frame number of the last occurrence or None if the track ID does not exist
        """

        if track_id in self.track_df["trackID"].values:
            return int(self.track_df.iloc[(self.track_df["trackID"] == track_id).argmax()]["last_frame"])
        else:
            return None

    def get_splitting_frame(self, track_id: int):
        """
        Returns the frame number of the splitting event if possible
        :param track_id: The track id of the cell
        :return: The frame number where the cell splits, if the cell does not split, None is returned
        """

        if np.any(self.track_df[self.track_df["trackID"] == track_id]["split"] == 1):
            current_selection = self.track_df[self.track_df["trackID"] == track_id]
            return int(current_selection.iloc[(current_selection["split"] == 1).argmax()]["frame"])
        else:
            return None

    def get_kids_id(self, track_id: int):
        """
        Returns the track ids of the daughter cells of a given track id
        :param track_id: The track ID of the mother cell
        :return: A tuple of ints containing the kids tracking ID if possible otherwise, None is returned
        """

        # get the kids
        d1_id = self.track_df.iloc[(self.track_df["trackID"] == track_id).argmax()]["trackID_d1"]
        d2_id = self.track_df.iloc[(self.track_df["trackID"] == track_id).argmax()]["trackID_d2"]

        if not np.isnan(d1_id) and not np.isnan(d2_id):
            return d1_id, d2_id
        else:
            return None

    def disconnect_lineage(self, track_id: int, frame_number: int):
        """
        Ends the lineage of a cell in a current frame (the lineage in all following cells will be changed)
        :param track_id: The track ID of the cell to disconnect
        :param frame_number: The last frame the cell should appear in
        :return: The new lineage and track IDs of the cell
        """

        # get and update the next ids
        new_track_id = self.next_track_id
        self.next_track_id += 1
        new_lineage_id = self.next_lineage_id
        self.next_lineage_id += 1

        # get the lineage ID of the cell
        old_lineage_id = self.track_df[self.track_df["trackID"] == track_id]["lineageID"].max()

        # update track and lineage id of the data frames
        self.track_df.loc[(self.track_df["trackID"] == track_id) &
                          (self.track_df["frame"] > frame_number), "trackID"] = new_track_id
        self.track_df.loc[(self.track_df["lineageID"] == old_lineage_id) &
                          (self.track_df["frame"] > frame_number), "lineageID"] = new_lineage_id

        # the new lineage gets a new first frame
        self.track_df.loc[self.track_df["trackID"] == new_track_id, "first_frame"] = frame_number + 1

        # update the last frame of the old cells (can use track id now)
        self.track_df.loc[self.track_df["trackID"] == track_id, "last_frame"] = frame_number

        # remove daughter cells from the previous lineage
        self.track_df.loc[self.track_df["trackID"] == track_id, "trackID_d1"] = np.nan
        self.track_df.loc[self.track_df["trackID"] == track_id, "trackID_d2"] = np.nan

        # cells that reference the old track ID as mother need to be updated as well
        self.track_df.loc[self.track_df["trackID_mother"] == track_id, "trackID_mother"] = new_track_id

        # add the transformation [first frame (inclusive9, old_id, new_id]
        self.track_id_transforms.append([frame_number + 1, track_id, new_track_id])

        # save everything to file
        self.track_df.to_csv(self.corrected_file, index=False)
        np.save(self.transformation_file, np.array(self.track_id_transforms))

        return new_lineage_id, new_track_id


@njit()
def update_labels(label_frame: np.ndarray, frame_number: int, transformations: np.ndarray):
    """
    Updates the labels in a given frame
    :param label_frame: The labels (2D array) of ints
    :param transformations: The transformations 2D array
    :return: The transformed labels
    """

    n, m = label_frame.shape
    n_transform = len(transformations)

    for i in range(n):
        for j in range(m):
            for k in range(n_transform):
                # check if relevant for frame
                if frame_number >= transformations[k, 0]:
                    if label_frame[i, j] == transformations[k, 1]:
                        label_frame[i, j] = transformations[k, 2]

    return label_frame


class CorrectionData(object):
    """
    This class handles all the data relevant for the tracking correction tool
    """

    def __init__(self, data_file: Union[str, bytes, os.PathLike], csv_file: Union[str, bytes, os.PathLike]):
        """
        Inits the class instance and reads the data, it will read already corrected data if it exists
        :param data_file: A hdf5 file containing the datasets "labels" and "images". The "labels" dataset has
                          three dimensions (TWH), has type int and contains the labels of the cells. The "images" is
                          a three dimensional array (TWH) containing the gray scale images in float type.
        :param csv_file: A csv_file containing the lineage data in MIDAP format
        """

        # we init the csv data
        self.tracking_data = TrackingData(csv_file=csv_file)

        # get the corrected name
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(f"The data file does not exists: {data_file}")
        if data_file.suffix.endswith(CORRECTION_SUFFIX):
            self.corrected_data_file = data_file
        else:
            self.corrected_data_file = data_file.with_suffix(data_file.suffix + CORRECTION_SUFFIX)

        # we open the h5 data as dask arrays
        if not self.corrected_data_file.exists():
            # copy the file to the corrected version for safety reason
            copyfile(src=data_file, dst=self.corrected_data_file)
        self.h5_file = h5py.File(self.corrected_data_file, "r")
        self.labels = da.from_array(self.h5_file["labels"])
        self.images = da.from_array(self.h5_file["images"])
        self.n_frames = len(self.images)

        # init the action stacks
        self.undo_stack = []
        self.redo_stack = []

    def get_image(self, frame_number: int):
        """
        Returns the image data of a given frame number.
        :param frame_number: The number of the frame
        :return: The image of the frame as numpy array
        """

        return self.images[frame_number].compute()

    def get_label(self, frame_number: int):
        """
        Returns the label data of a given frame number.
        :param frame_number: The number of the frame
        :return: The label of the frame as numpy array
        """

        # if there is nothing to transform
        if len(self.tracking_data.track_id_transforms) == 0:
            return self.labels[frame_number].compute()
        else:
            # update the labels
            return update_labels(self.labels[frame_number].compute(),
                                 frame_number,
                                 np.asarray(self.tracking_data.track_id_transforms, dtype=np.int32))

    def get_selection(self, frame_number: int, selection: Optional[int], mark_orphans: bool, mark_dying: bool):
        """
        Returns the selection data of a given frame number.
        :param frame_number: The number of the frame
        :param selection: The ID of the cell to select
        :param mark_orphans: Whether orphans should be marked
        :param mark_dying: Whether dying cells should be marked
        :return: The selection of the frame as numpy array
        """

        # get the label
        label = self.get_label(frame_number=frame_number)

        # init the new data
        new_data = np.zeros_like(label)

        # orphan selection
        if mark_orphans and frame_number != 0:
            orphan_ids = self.tracking_data.get_number_of_orphans(frame_number=frame_number, return_ids=True)
            for orphan_id in orphan_ids:
                new_data = np.where(label == orphan_id, 3, new_data)

        # dying selection
        if mark_dying and frame_number != self.n_frames - 1:
            dying_ids = self.tracking_data.get_number_of_dying(frame_number=frame_number, return_ids=True)
            for dying_id in dying_ids:
                new_data = np.where(label == dying_id, 4, new_data)

        # we do this here to overwrite the other data
        if selection is not None:
            # normal selection
            new_data = np.where(label == selection, 1, new_data)

            # kids selections
            daughters = self.tracking_data.get_kids_id(track_id=selection)
            if daughters is not None:
                new_data = np.where(label == daughters[0], 2, new_data)
                new_data = np.where(label == daughters[1], 2, new_data)

        return new_data

    def disconnect_lineage(self, track_id: int, frame_number: int):
        """
        Ends the lineage of a cell in a current frame (the lineage in all following cells will be changed)
        :param track_id: The track ID of the cell to disconnect
        :param frame_number: The last frame the cell should appear in
        """

        # we disconnect the linage in the data
        new_lineage_id, new_track_id = self.tracking_data.disconnect_lineage(track_id=track_id,
                                                                             frame_number=frame_number)
        show_info(f"Disconnected lineage of cell {track_id} in frame {frame_number}!")

    def __enter__(self):
        """
        The context manager enter method
        :return: The instance of the class
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The teardown mechanism of the context manager
        :param exc_type: The Exception type, can be None
        :param exc_val: The Exception value, can be None
        :param exc_tb: The trace back
        :return: If there was an exception the method returns True if the exception was handled gracefully, otherwise
                 we do the teardown and the exception is forwarded
        """

        # close the h5 file
        self.h5_file.close()
