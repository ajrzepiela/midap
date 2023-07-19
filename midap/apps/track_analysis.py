import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Union
from skimage.measure import regionprops_table

def open_h5(h5_file: Union[str,os.PathLike]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Opens h5 file
    """
    f = h5py.File(h5_file, 'r')
    images = np.array(f['images'])
    labels = np.array(f['labels'])
    return images, labels

def main(path, channels, tracking_class):
    # generate pathnames
    filename_h5 = 'tracking_' + tracking_class.lower() + '.h5'
    filename_csv = 'track_output_' + tracking_class.lower() + '.csv'
    path_ref_h5 = path.joinpath(channels[0], 'track_output', filename_h5)
    path_ref_csv = path.joinpath(channels[0], 'track_output', filename_csv)
    paths_fluo_h5 = [path.joinpath(c, 'track_output', filename_h5) for c in channels[1:]]

    # load h5 files
    _, labels_ref = open_h5(path_ref_h5)
    images_fluo = np.array([open_h5(pf)[0] for pf in paths_fluo_h5])
    labels_fluo = np.array([open_h5(pf)[1] for pf in paths_fluo_h5])

    # load phase tracking dataframe and add columns
    track_output_ref = pd.read_csv(path_ref_csv, index_col='Unnamed: 0')
    track_output_fluo_change = track_output_ref.copy()
    new_columns = ['intensity_' + c for c in channels[1:]]
    for nc in new_columns:
        track_output_fluo_change[nc] = np.nan
   
    # loop through time frames and add mean intensities
    time_frames = len(labels_ref)
    for t in range(time_frames):
        props_ref = regionprops_table(labels_ref[t], properties=['label', 'coords'])
        props_fluo = [regionprops_table(l, intensity_image=i, properties=['label', 'coords', 'intensity_mean', 'image_intensity']) for l,i in zip(labels_fluo[:,t,:,:], images_fluo[:,t,:,:])]

        df_ref = pd.DataFrame(props_ref, index=props_ref['label'])
        df_fluo = [pd.DataFrame(pf,index=pf['label']) for pf in props_fluo]

        # loop through cells
        for l in df_ref.label:
            coords = df_ref.loc[l].coords
            mean_intensities = np.mean(images_fluo[:,t,coords[:,0],coords[:,1]], axis=1)
            
            for nc, mi in zip(new_columns, mean_intensities):
                track_output_fluo_change.loc[(track_output_fluo_change.trackID==l) & (track_output_fluo_change.frame==t), nc] = mi

    # save new output
    filename_csv_new = Path(Path(filename_csv).stem + '_fluo_change' + Path(filename_csv).suffix)
    track_output_fluo_change.to_csv(path.joinpath(channels[0], 'track_output', filename_csv_new))  
        

if __name__ == "__main__":
    path = Path('/Users/franziskaoschmann/Documents/midap-datasets/Simon-GlucoseAcetate/GlucoseAcetateExampleData/Data/pos1')
    channels = ['ph', 'gfp', 'mCherry']
    tracking_class = 'STrack'
    main(path, channels, tracking_class=tracking_class)