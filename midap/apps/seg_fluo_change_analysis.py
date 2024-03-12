import numpy as np
import os
from pathlib import Path
from typing import Union, List

from skimage import io
from skimage.measure import regionprops_table

import pandas as pd
from tqdm import tqdm


def load_img_stack(path: Union[str, os.PathLike], files: List[Union[str, os.PathLike]]):
    """
    Loads all imgs from folder and combines them to stack.
    :param path: Path to channel folder.
    :param files: List with file names of images.
    """
    stack = []
    for f in files:
        stack.append(io.imread(path.joinpath(f)))
    stack = np.array(stack)
    return stack


def fluo_analysis_per_channel(
    path: Union[str, os.PathLike], ref_channel: str, add_channel: str
):
    """
    Loads tracking output and adds intensities per cell and channel to dataframe.
    :param path: Path to output folder.
    :param channels: List with channels.
    :param tracking_class: Name of used tracking class.
    """

    # create path's
    path_ref_channel = Path(path).joinpath(ref_channel)
    path_add_channel = Path(path).joinpath(add_channel)

    path_ref_ch_seg = path_ref_channel.joinpath("seg_im")
    path_add_ch_img = path_add_channel.joinpath("cut_im")
    path_add_ch_img_raw = path_add_channel.joinpath("cut_im_rawcounts")

    # get all file names from folder
    ref_ch_seg_all_files = np.sort(os.listdir(path_ref_ch_seg))
    add_ch_img_all_files = np.sort(os.listdir(path_add_ch_img))
    add_ch_img_raw_all_files = np.sort(os.listdir(path_add_ch_img_raw))

    # load img stacks
    segs_ref_ch = load_img_stack(path_ref_ch_seg, ref_ch_seg_all_files)
    img_add_ch = load_img_stack(path_add_ch_img, add_ch_img_all_files)
    img_add_ch_raw = load_img_stack(path_add_ch_img_raw, add_ch_img_raw_all_files)

    # Loop through all frames
    df_all = pd.DataFrame()

    for frame in tqdm(range(len(segs_ref_ch))):
        ref_ch = segs_ref_ch[frame]
        add_ch = img_add_ch[frame]
        add_ch_raw = img_add_ch_raw[frame]

        props = regionprops_table(
            ref_ch,
            intensity_image=add_ch_raw,
            properties=(
                "label",
                "coords",
                "area",
                "bbox",
                "intensity_max",
                "intensity_mean",
                "intensity_min",
                "minor_axis_length",
                "major_axis_length",
                "centroid",
            ),
        )

        df = pd.DataFrame(props)

        df = df.set_index("label")

        intensities_add_ch = []
        intensities_raw_add_ch = []

        # Loop through all cells of frame
        for i in df.index:
            row, col = df.loc[i].coords.T
            intensities_add_ch.append(np.mean(add_ch[row, col]))
            intensities_raw_add_ch.append(np.mean(add_ch_raw[row, col]))

        df["intensity_" + add_channel] = intensities_add_ch
        df["intensity_raw_" + add_channel] = intensities_raw_add_ch
        df["frame_number"] = [frame] * len(df.index)

        df = df.rename(
            columns={
                "bbox-0": "min_row",
                "bbox-1": "min_col",
                "bbox-2": "max_row",
                "bbox-3": "max_col",
                "centroid-1": "x",
                "centroid-0": "y",
            }
        )

        df.drop(["coords"], axis=1, inplace=True)

        df_all = pd.concat([df_all, df])

    return df_all


def main(path: Union[str, os.PathLike], channels: List[str]):
    """
    Loads tracking output and adds intensities per cell and channel to dataframe.
    :param path: Path to output folder.
    :param channels: List with channels.
    :param tracking_class: Name of used tracking class.
    """
    add_channels = channels[1:]

    df_all_channels = pd.DataFrame()
    for add_channel in add_channels:
        df = fluo_analysis_per_channel(
            path=path, ref_channel=channels[0], add_channel=add_channel
        )
        df_all_channels = pd.concat([df_all_channels, df], axis=1)

    df_all_channels = df_all_channels.loc[
        :, ~df_all_channels.columns.duplicated()
    ].copy()

    path_ref_channel = Path(path).joinpath(channels[0])
    df_all_channels.to_csv(path_ref_channel.joinpath("fluo_intensities.csv"))


if __name__ == "__main__":
    path = Path(
        "/Users/franziskaoschmann/Documents/midap-datasets/fluo_change_giulia/Image_Analysis/pos0/"
    )
    channels = ["phase", "mcherry"]
    tracking_class = "STrack"
    main(path, channels)
