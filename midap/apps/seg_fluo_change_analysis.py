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

def main(path: Union[str, os.PathLike], channels: List[str]):
    """
    Loads tracking output and adds intensities per cell and channel to dataframe.
    :param path: Path to output folder.
    :param channels: List with channels.
    :param tracking_class: Name of used tracking class.
    """

    path_ph_channel = Path(path).joinpath(channels[0])
    path_ch1_channel = Path(path).joinpath(channels[1])
    path_ch2_channel = Path(path).joinpath(channels[2])

    path_ph_seg = path_ph_channel.joinpath('seg_im')
    path_ch1_img = path_ch1_channel.joinpath('cut_im')
    path_ch2_img = path_ch2_channel.joinpath('cut_im')
    path_ch1_img_raw = path_ch1_channel.joinpath('cut_im_rawcounts')
    path_ch2_img_raw = path_ch2_channel.joinpath('cut_im_rawcounts')

    ph_seg_all_files = np.sort(os.listdir(path_ph_seg))
    ch1_img_all_files = np.sort(os.listdir(path_ch1_img))
    ch2_img_all_files = np.sort(os.listdir(path_ch2_img))

    ch1_img_raw_all_files = np.sort(os.listdir(path_ch1_img_raw))
    ch2_img_raw_all_files = np.sort(os.listdir(path_ch2_img_raw))

    segs_ph = load_img_stack(path_ph_seg, ph_seg_all_files)
    img_ch1 = load_img_stack(path_ch1_img, ch1_img_all_files)
    img_ch2 = load_img_stack(path_ch2_img, ch2_img_all_files)
    img_ch1_raw = load_img_stack(path_ch1_img_raw, ch1_img_raw_all_files)
    img_ch2_raw = load_img_stack(path_ch2_img_raw, ch2_img_raw_all_files)


    # Loop through all frames
    df_all = pd.DataFrame()

    for frame in tqdm(range(len(segs_ph))):
        ph = segs_ph[frame]
        ch1 = img_ch1[frame]
        ch2 = img_ch2[frame]
        ch1_raw = img_ch1_raw[frame]
        ch2_raw = img_ch2_raw[frame]

        props = regionprops_table(ph, properties=('label', 'coords'))
    
        df = pd.DataFrame(props)

        df = df.set_index('label')

        intensities_ch1 = []
        intensities_ch2 = []
        intensities_raw_ch1 = []
        intensities_raw_ch2 = []

        # Loop through all cells of frame
        for i in df.index:
            intensities_ch1.append(np.mean(ch1[df.loc[i].coords]))
            intensities_ch2.append(np.mean(ch2[df.loc[i].coords]))
            intensities_raw_ch1.append(np.mean(ch1_raw[df.loc[i].coords]))
            intensities_raw_ch2.append(np.mean(ch2_raw[df.loc[i].coords]))

        df['intensity_'+channels[1]] = intensities_ch1
        df['intensity_'+channels[2]] = intensities_ch2
        df['intensity_raw_'+channels[1]] = intensities_raw_ch1
        df['intensity_raw_'+channels[2]] = intensities_raw_ch2
        df['frame_number'] = [frame]*len(df.index)
        df.drop(['coords'], axis=1, inplace=True)

        df_all = pd.concat([df_all, df])

    df_all.to_csv(path_ph_channel.joinpath('fluo_intensities.csv'))


if __name__ == "__main__":
    path = Path(
        "/Users/franziskaoschmann/Documents/midap-datasets/Simon-GlucoseAcetate/GlucoseAcetateExampleData/Data/pos1"
    )
    channels = ["ph", "gfp", "mCherry"]
    tracking_class = "STrack"
    main(path, channels)
