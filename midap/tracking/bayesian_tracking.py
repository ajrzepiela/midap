import os
from pathlib import Path
from typing import Union

import btrack
import h5py
import numpy as np
import pandas as pd
from btrack.constants import BayesianUpdates
from numba import njit, typed, types
from tqdm import tqdm

from .base_tracking import Tracking


@njit(cache=True)
def label_transform(labels: np.ndarray, transformations: np.ndarray):
    """
    Transforms the labels of a labelled image according to the transformations
    :param labels: A 3D array TWH of type int32 containing the labels
    :param transformations: A array of shape (N, 3) of type int32 containing the transformations to apply. The
                            transformations are of the form (frame, old, new)
    """

    # extract shapes
    t, n, m = labels.shape
    n_transforms = len(transformations)

    for t_step in range(t):
        # build the transformations dict
        trans_dict = typed.Dict.empty(key_type=types.int32, value_type=types.int32)
        for i in range(n_transforms):
            if transformations[i, 0] == t_step:
                trans_dict[transformations[i, 1]] = transformations[i, 2]

        # apply transformations
        for i in range(n):
            for j in range(m):
                if labels[t_step, i, j] in trans_dict:
                    labels[t_step, i, j] = trans_dict[labels[t_step, i, j]]


class BayesianCellTracking(Tracking):
    """
    A class for cell tracking using Bayesian tracking
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DeltaV2Tracking using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the baseclass init
        """

        # base class init
        super().__init__(*args, **kwargs)

        # read the files
        raws = []
        segs = []
        for i in range(self.num_time_steps):
            r, _, s, _ = self.load_data(i, label=True)
            raws.append(r)
            segs.append(s)
        self.seg_imgs = np.array(segs)
        self.raw_imgs = np.array(raws)

    def track_all_frames(self, output_folder: Union[str, bytes, os.PathLike]):
        """
        Tracks all frames and converts output to standard format.
        :param output_folder: Folder for the output
        """

        tracks = self.run_model()
        df, label_stack = self.generate_midap_output(tracks=tracks)
        data_file, csv_file = self.store_lineages(output_folder=output_folder, df=df, label_stack=label_stack)

        return data_file, csv_file

    def run_model(self):
        """
        Run Bayesian model.
        """

        # gen the inputs
        objects = btrack.utils.segmentation_to_objects(segmentation=(self.seg_imgs).astype(int), intensity_image=self.raw_imgs,
                                                       assign_class_ID=True)
        config_file = Path(__file__).parent.joinpath("btrack_conf.json")

        # choose update method depending on number of cells
        cum_sum_cells = np.sum([len(np.unique(s)) - 1 for s in self.seg_imgs])
        num_frames = len(self.seg_imgs)
        max_cells_frame = 1_000
        max_cells_total = num_frames * max_cells_frame

        # initialise a tracker session using a context manager
        with btrack.BayesianTracker() as tracker:
            if cum_sum_cells < max_cells_total:
                tracker.update_method = BayesianUpdates.EXACT
            else:
                tracker.update_method = BayesianUpdates.APPROXIMATE
                tracker.max_search_radius = 256

            # configure the tracker using a config file
            tracker.configure(config_file)

            # set params
            tracker.tracking_updates = ["VISUAL", "MOTION"]

            # append the objects to be tracked
            tracker.append(objects)

            # track them (in interactive mode)
            tracker.track(step_size=100)

            # generate hypotheses and run the global optimizer
            tracker.optimize()

            # get the tracks as a python list
            tracks = tracker.tracks

        return tracks

    def generate_midap_output(self, tracks):
        """
        Generate label stack based on tracking output.
        :param tracks: The tracks generated from btrack
        :return: The midap dataframe and labelstack
        """

        self.logger.info("Creating data frame...")
        # init the dataframe
        columns = ['frame', 'labelID', 'trackID', 'lineageID', 'trackID_d1', 'trackID_d2', 'split',
                   'trackID_mother', 'first_frame', 'last_frame']
        df = pd.DataFrame(columns=columns)

        # list to transform the labels later
        label_transforms = []
        global_id = 1
        lineage_id = 1
        for track in tqdm(tracks):
            # set the parent id
            parent_id = None

            # go through steps
            steps = track["t"]
            for i, t in enumerate(steps):
                # skip dummies
                if track["dummy"][i]:
                    continue

                df.loc[global_id, "frame"] = t
                df.loc[global_id, "labelID"] = int(track["class_id"][i])
                df.loc[global_id, "trackID"] = track["ID"]
                df.loc[global_id, "first_frame"] = min(steps)
                df.loc[global_id, "last_frame"] = max(steps)
                df.loc[global_id, "split"] = 0

                # check if there is a parent
                if track["parent"] != track["ID"]:
                    parent_id = track["parent"]
                    df.loc[global_id, "lineageID"] = df.loc[df["trackID"] == parent_id, "lineageID"].max()
                    df.loc[global_id, "trackID_mother"] = df.loc[df["trackID"] == parent_id, "trackID"].max()
                else:
                    df.loc[global_id, "lineageID"] = lineage_id

                # add the transformation
                label_transforms.append([t, int(track["class_id"][i]), track["ID"]])

                # increment global ID
                global_id += 1

            # update the parent
            if parent_id is not None:
                # set the split event
                last_frame = df.loc[df["trackID"] == parent_id, "last_frame"].max()
                df.loc[(df["trackID"] == parent_id) & (df["frame"] == last_frame), "split"] = 1
                # assign kids
                if df.loc[(df["trackID"] == parent_id), "trackID_d1"].isna().all():
                    df.loc[(df["trackID"] == parent_id), "trackID_d1"] = track["ID"]
                elif df.loc[(df["trackID"] == parent_id), "trackID_d2"].isna().all():
                    df.loc[(df["trackID"] == parent_id), "trackID_d2"] = track["ID"]
                else:
                    raise ValueError(f"Cell with trackID {parent_id} splits into more than 2 cells!")

            # increment lineage
            else:
                lineage_id += 1

        self.logger.info("Creating label stack...")
        # Note: There is the function btrack.utils.update_segmentation that deos this as well, however, this function
        # removes all segmentations whose centroid is not inside the cell, so we use the class_id work around
        label_stack = self.seg_imgs.copy().astype(np.int32)
        label_transform(labels=label_stack, transformations=np.array(label_transforms, dtype=np.int32))

        return df, label_stack

    def store_lineages(self, output_folder: str, df: pd.DataFrame, label_stack: np.ndarray):
        """
        Store tracking output files: labeled stack, tracking output, input files.
        :param output_folder: Folder where to store the data
        :param df: The pandas data frame to store as csv
        :param label_stack: The labelstack array to store
        :return: The data file name and csv file name
        """

        # transform to path
        output_folder = Path(output_folder)

        # save everything
        csv_file = output_folder.joinpath("track_output_bayesian.csv")
        df.to_csv(csv_file, index=True, index_label="globalID")

        data_file = output_folder.joinpath("tracking_bayesian.h5")
        with h5py.File(data_file, "w") as hf:
            hf.create_dataset("images", data=self.raw_imgs.astype(float), dtype=float)
            hf.create_dataset("labels", data=label_stack.astype(int), dtype=int)

        with h5py.File(output_folder.joinpath("segmentations_bayesian.h5"), "w") as hf:
            hf.create_dataset("segmentations", data=self.seg_imgs)

        return data_file, csv_file
