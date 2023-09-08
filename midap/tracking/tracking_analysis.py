import glob
import h5py
import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, List
from skimage.measure import regionprops_table
from skimage import io

class FluoChangeAnalysis:
    def __init__(self, path, channels, tracking_class) -> None:
        """
        Inits params.
        :param path: Path to output folder.
        :param channels: List with channels.
        :param tracking_class: Name of used tracking class.
        """
        self.path = path
        self.channels = channels
        self.tracking_class = tracking_class

    def gen_pathnames(self):
        """
        Generates pathnames based on tracking class and path to output folder.
        """
        filename_h5 = "tracking_" + self.tracking_class.lower() + ".h5"
        self.filename_csv = "track_output_" + self.tracking_class.lower() + ".csv"
        self.path_ref_h5 = self.path.joinpath(
            self.channels[0], "track_output", filename_h5
        )
        self.path_ref_csv = self.path.joinpath(
            self.channels[0], "track_output", self.filename_csv
        )
        self.paths_fluo_h5 = [
            self.path.joinpath(c, "track_output", filename_h5)
            for c in self.channels[1:]
        ]
        self.path_fluo_png = [
            self.path.joinpath(c, "cut_im_rawcounts") for c in self.channels[1:]
        ]

    def load_images(self):
        """
        Loads images from tracking output.
        """
        # load h5 files
        _, self.labels_ref = self.open_h5(self.path_ref_h5)
        self.images_fluo = np.array([self.open_h5(pf)[0] for pf in self.paths_fluo_h5])

        # load raw count images
        self.images_fluo_raw = np.array(
            [self.open_img_folder(pf, "tif") for pf in self.path_fluo_png]
        )

    def open_h5(
        self, h5_file: Union[str, os.PathLike]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Opens h5 file
        :param h5_file: Path to h5-file.
        """
        f = h5py.File(h5_file, "r")
        images = np.array(f["images"])
        labels = np.array(f["labels"])
        return images, labels

    def open_img_folder(self, path: Union[str, os.PathLike], ext: str) -> np.ndarray:
        """
        Opens all images from given folder.
        :param path: Path to folder containing images.
        :param ext: File extension of images.
        """
        file_names = np.sort(glob.glob(str(path) + "/*." + ext))
        img_array = np.array([io.imread(f) for f in file_names])
        return img_array

    def gen_column_names(self):
        """
        Generates new column names for fluo change.
        """
        self.new_columns = ["mean_norm_intensity_" + c for c in self.channels[1:]]
        self.new_columns_raw = ["mean_raw_intensity_" + c for c in self.channels[1:]]

    def create_output_df(self, path_ref_csv: Union[str, os.PathLike]) -> pd.DataFrame:
        """
        Loads tracking output and adds two more columns per cell for the normalized
        and raw intensities per fluorescence channel.
        :param path_ref_csv: Path to csv where fluo intensity should be added.
        """
        track_output_ref = pd.read_csv(path_ref_csv, index_col="Unnamed: 0")
        df_fluo_change = track_output_ref.copy()

        for nc in self.new_columns:
            df_fluo_change[nc] = np.nan
        for nc in self.new_columns_raw:
            df_fluo_change[nc] = np.nan

        return df_fluo_change

    def add_fluo_intensity(self, df_fluo_change: pd.DataFrame) -> pd.DataFrame:
        """
        Loops through time frames and adds mean intensities.
        :param df_fluo_change: Tracking output with additional columns for fluo intensity.
        """
        time_frames = len(self.labels_ref)
        for t in range(time_frames):
            props_ref = regionprops_table(
                self.labels_ref[t], properties=["label", "coords"]
            )

            df_ref = pd.DataFrame(props_ref, index=props_ref["label"])

            # loop through cells
            for l in df_ref.label:
                coords = df_ref.loc[l].coords
                mean_intensities = np.mean(
                    self.images_fluo[:, t, coords[:, 0], coords[:, 1]], axis=1
                )
                mean_intensities_raw = np.mean(
                    self.images_fluo_raw[:, t, coords[:, 0], coords[:, 1]], axis=1
                )

                for nc, mi in zip(self.new_columns, mean_intensities):
                    df_fluo_change.loc[
                        (df_fluo_change.trackID == l) & (df_fluo_change.frame == t), nc
                    ] = mi

                for nc, mi in zip(self.new_columns_raw, mean_intensities_raw):
                    df_fluo_change.loc[
                        (df_fluo_change.trackID == l) & (df_fluo_change.frame == t), nc
                    ] = mi

        return df_fluo_change

    def save_fluo_change(self, df_fluo_change: pd.DataFrame):
        """
        Saves new output.
        :param df_fluo_change: Tracking output with additional columns for fluo intensity.
        """
        filename_csv_new = Path(
            Path(self.filename_csv).stem
            + "_fluo_change"
            + Path(self.filename_csv).suffix
        )
        df_fluo_change.to_csv(
            self.path.joinpath(self.channels[0], "track_output", filename_csv_new)
        )
