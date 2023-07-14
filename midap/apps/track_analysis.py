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

def add_labels(labels_ph: np.ndarray, labels_gfp: np.ndarray, labels_mcherry: np.ndarray) -> np.ndarray:
    """
    Superimposes labels
    """
    added_labels = (labels_ph > 0).astype(int)
    added_labels += 2*(labels_gfp > 0).astype(int)
    added_labels += 4*(labels_mcherry > 0).astype(int)

    return added_labels

def save_added_labels(added_labels):
    """
    Saves added labels as h5 file
    """
    path = '../../../midap-datasets/Simon-GlucoseAcetate/GlucoseAcetateExampleData/Data/pos1/ph/track_output/'
    hf = h5py.File(path+'tracking_postprocessed.h5', 'w')
    hf.create_dataset('added_labels', data=added_labels)
    hf.close()
    # fig, ax = plt.subplots()
    # cax = ax.imshow(added_labels[10])
    # cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3, 4, 5, 4, 6, 7])
    # # cbar.ax.set_yticklabels(['BG', 'PH', 'GFP', 'mCherry', 'PH+GFP', 'PH+mCherry' + 'GFP+mCherry', 'PH+GFP+mCherry'])

    # # fig.savefig('../../../midap-datasets/Simon-GlucoseAcetate/GlucoseAcetateExampleData/Data/pos1/ph/track_output/added_labels.png')

def get_num2fluo():
    num2fluo = dict()

    num2fluo[1] = ['ph']
    num2fluo[2] = ['gfp']
    num2fluo[4] = ['mcherry']
    num2fluo[3] = ['ph','gfp']
    num2fluo[5] = ['ph','mcherry']
    num2fluo[6] = ['gfp','mcherry']
    num2fluo[7] = ['ph','gfp','mcherry']

    return num2fluo

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
    path = PosixPath('/Users/franziskaoschmann/Documents/midap-datasets/Simon-GlucoseAcetate/GlucoseAcetateExampleData/Data/pos1')
    channels = ['ph', 'gfp', 'mCherry']
    tracking_class = 'STrack'
    main(path, channels, tracking_class=tracking_class)