import os
from pathlib import Path
from typing import Union, List

import btrack
import h5py
import numpy as np
import pandas as pd
from btrack import datasets
from btrack.constants import BayesianUpdates
from scipy.spatial import distance
from skimage.measure import label, regionprops

from .base_tracking import Tracking


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
        self.seg_imgs = np.array([label(self.load_data(cur_frame)[2]) for cur_frame in range(0, self.num_time_steps)])
        self.raw_imgs = np.array([self.load_data(cur_frame)[0] for cur_frame in range(0, self.num_time_steps)])

    def track_all_frames(self, output_folder: Union[str, bytes, os.PathLike]):
        """
        Tracks all frames and converts output to standard format.
        :param output_folder: Folder for the output
        """

        tracks = self.run_model()
        track_output = self.convert_data(tracks=tracks)
        label_stack = self.generate_label_stack(tracks=tracks)
        label_stack_correct, track_output_correct = self.correct_label_stack(label_stack, track_output)
        data_file, csv_file = self.store_lineages(output_folder=output_folder, track_output=track_output_correct,
                                                  label_stack_correct=label_stack_correct)

        return data_file, csv_file

    def run_model(self):
        """
        Run Bayesian model.
        """

        # gen the inputs
        objects = btrack.utils.segmentation_to_objects(segmentation=self.seg_imgs, intensity_image=self.raw_imgs,
                                                       assign_class_ID=True)
        config_file = datasets.cell_config()

        # choose update method depending on number of cells
        cum_sum_cells = np.sum([np.max(s) for s in self.seg_imgs])
        num_frames = len(self.seg_imgs)
        max_cells_frame = 1_000
        max_cells_total = num_frames * max_cells_frame

        if cum_sum_cells < max_cells_total:
            update_method = BayesianUpdates.EXACT
        else:
            update_method = BayesianUpdates.APPROXIMATE

        # initialise a tracker session using a context manager
        with btrack.BayesianTracker() as tracker:
            tracker.update_method = update_method

            # configure the tracker using a config file
            tracker.configure(config_file)

            # set params
            tracker.max_search_radius = 100
            tracker.tracking_updates = ["VISUAL", "MOTION"]

            # append the objects to be tracked
            tracker.append(objects)

            # track them (in interactive mode)
            tracker.track(step_size=100)

            # generate hypotheses and run the global optimizer
            tracker.optimize()

            # get the tracks as a python list
            tracks = tracker.tracks

        self.logger.info("Creating label stack...")
        # init the dataframe
        columns = ['frame', 'labelID', 'trackID', 'lineageID', 'trackID_d1', 'trackID_d2', 'split',
                   'trackID_mother', 'first_frame', 'last_frame']
        df = pd.DataFrame(columns=columns)

        # list to transform the labels later
        label_transforms = []
        global_id = 1
        for track in tracks:

            global_id += 1



        return tracks

    def generate_label_stack(self, tracks):
        """
        Generate label stack based on tracking output.
        """

        label_stack = np.zeros(self.seg_imgs.shape)

        for tr in tracks:
            for i, t in enumerate(tr["t"]):
                
                # get coords from labaled segmentations
                centroid = (tr['y'][i], tr['x'][i])
                coords = self.find_coords(centroid, regionprops(self.seg_imgs[t]))
                row_coord = coords[:, 0].astype(int)
                col_coord = coords[:, 1].astype(int)

                label_stack[t][row_coord, col_coord] = tr["ID"]

        return label_stack

    def find_coords(self, point: tuple, props: List):
        """
        Find coordinates for cell based on centrtoid.
        :point: Center point of tracked cell
        :seg: Segmentation image
        """
        centroids = [r.centroid for r in props]
        coords = [r.coords for r in props]
        ix_cell = np.argsort([distance.euclidean(c, point) for c in centroids])[0]
        return coords[ix_cell]

    def find_nearest_neighbour(self, point: tuple, props: List):
        """
        Find nearest neighboring cell in segmentation image.
        :point: Center point of tracked cell
        :seg: Segmentation image
        """
        centroids = [r.centroid for r in props]
        ix_min = np.argsort([distance.euclidean(c, point) for c in centroids])[1]
        return ix_min

    def find_mother(self, point: tuple, props: List):
        """
        Find nearest neighboring cell in segmentation image.
        :point: Center point of tracked cell
        :seg: Segmentation image
        """
        centroids = [r.centroid for r in props]
        ix_min = np.argsort([distance.euclidean(c, point) for c in centroids])[0]
        return ix_min

    def correct_label_stack(self, label_stack, track_output):
        """
        Correct label_stack and track_output to fit to community standard:
        - new label for mother after cell split
        - add IDs of daughter cells
        """

        label_stack_correct = label_stack
        track_output_correct = track_output.copy()

        track_output_correct["trackID_d1"] = track_output_correct["trackID"]
        track_output_correct["trackID_d2"] = track_output_correct["trackID"]
        track_output_correct["trackID_mother"] = track_output_correct["trackID"]

        for t in range(1, len(label_stack)):

            # find new IDs
            labels_prev_frame = set(np.unique(label_stack[t - 1]))
            labels_cur_frame = np.unique(label_stack[t])
            diff_ix = np.array([l2 not in labels_prev_frame for l2 in labels_cur_frame])

            # find labels and centroids of new cells in current time frame and of potential mother cells in prev frame
            reg_prev = regionprops((label_stack[t - 1]).astype(int))
            reg_cur = regionprops((label_stack[t]).astype(int))
            centroids_cur = [r.centroid for r in reg_cur]

            labels = [r.label for r in reg_cur]
            labels_prev = [r.label for r in reg_prev]
            new_cells = labels_cur_frame[diff_ix]

            # loop over new cells find closest cell and do correction of label stack
            for c in new_cells:
                # find closest neighbouring cell in current and prev time frame
                ix = np.where(labels == c)[0][0]
                ix_closest_cell = self.find_nearest_neighbour(centroids_cur[ix], reg_cur)
                ix_mother_cell = self.find_mother(centroids_cur[ix], reg_prev)

                # set new IDs of daughter and mother cells
                new_ID_d1 = int(labels[ix])
                new_ID_d2 = labels[ix_closest_cell]
                mother = labels_prev[ix_mother_cell]
                label_stack_correct[t:][label_stack[t:] == mother] = new_ID_d2

                # correct df
                ix_col_mother = np.where(track_output_correct.columns == "trackID_mother")[0][0]
                ix_col_ID_d1 = np.where(track_output_correct.columns == "trackID_d1")[0][0]
                ix_col_ID_d2 = np.where(track_output_correct.columns == "trackID_d2")[0][0]

                for t_tmp_1 in range(0, t):

                    filter_t = track_output_correct["frame"] == t_tmp_1
                    filter_ID = track_output_correct["trackID"] == mother

                    # set daughter IDs in all prev frames
                    try:
                        ix_cell = np.where(filter_t & filter_ID)[0][0]
                        track_output_correct.iloc[ix_cell, ix_col_ID_d1] = new_ID_d1
                        track_output_correct.iloc[ix_cell, ix_col_ID_d2] = new_ID_d2

                    except IndexError:  # if cell skips frame
                        pass

                max_t = track_output_correct[
                    track_output_correct["trackID"] == new_ID_d1
                ]["frame"].max()
                for t_tmp_2 in range(t, max_t + 1):  # +1

                    filter_t = track_output_correct["frame"] == t_tmp_2
                    filter_ID = track_output_correct["trackID"] == mother
                    filter_ID_d1 = track_output_correct["trackID"] == new_ID_d1
                    filter_ID_d2 = track_output_correct["trackID"] == new_ID_d2

                    # set mother ID for d1 and d2
                    try:
                        ix_d1 = np.where(filter_t & filter_ID_d1)[0][0]
                        track_output_correct.iat[ix_d1, ix_col_mother] = mother
                        ix_d2 = np.where(filter_t & filter_ID_d2)[0][0]
                        track_output_correct.iat[ix_d2, ix_col_mother] = mother

                    except IndexError:  # if cell skips frame
                        pass

        return label_stack_correct, track_output_correct

    def __new_ID(self):
        return np.max(self.label_stack_correct) + 1

    def convert_data(self, tracks):
        """
        Convert tracking output into dataframe in standard format.
        """

        # generate subset of dict
        keys_to_extract = [
            "t",
            "ID",
        ]
        cells = []
        for cell in tracks:
            tmp_dict = cell.to_dict()
            tmp_dict["first_frame"] = tmp_dict["t"][0]
            tmp_dict["last_frame"] = tmp_dict["t"][-1]
            cell_dict = {key: tmp_dict[key] for key in keys_to_extract}

            cells.append(cell_dict)

        # transform subset into df
        track_output = pd.DataFrame(cells[0])
        for c in cells[1:]:
            df_cells_old = track_output
            df_cells_new = pd.DataFrame(c)
            track_output = pd.concat([df_cells_old, df_cells_new])

        track_output.sort_values(by="t", inplace=True)
        track_output.rename(columns={"t": "frame", "ID": "trackID"}, inplace=True)

        return track_output

    def store_lineages(self, output_folder: str, track_output: pd.DataFrame, label_stack_correct: np.ndarray):
        """
        Store tracking output files: labeled stack, tracking output, input files.
        :output_folder: Folder where to store the data
        """

        # transform to path
        output_folder = Path(output_folder)

        # save everything
        csv_file = output_folder.joinpath("track_output_bayesian.csv")
        track_output.to_csv(csv_file, index=True)

        data_file = output_folder.joinpath("tracking_bayesian.h5")
        with h5py.File(data_file, "w") as hf:
            hf.create_dataset("images", data=self.raw_imgs.astype(float), dtype=float)
            hf.create_dataset("labels", data=label_stack_correct.astype(int), dtype=int)

        with h5py.File(output_folder.joinpath("segmentations_bayesian.h5"), "w") as hf:
            hf.create_dataset("segmentations", data=self.seg_imgs)

        return data_file, csv_file

