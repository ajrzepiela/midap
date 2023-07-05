import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Union

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

def main(file_ph, file_gfp, file_mcherry):

    # load h5 files
    images_ph, labels_ph = open_h5(file_ph)
    images_gfp, labels_gfp = open_h5(file_gfp)
    images_mcherry, labels_mcherry = open_h5(file_mcherry)

    # superimpose labels with colorcode
    added_labels = add_labels(labels_ph, labels_gfp, labels_mcherry)

    # save colorcoded suerposition
    save_added_labels(added_labels)

    # check per cell in phase if cell is labeled (ph=1, gfp=2, mcherry=4, gfp+ph = 3, mcherry+ph = 5, gfp+mcherry=6, gfp+mcherry+ph=7)
    # Here example for one cell
    # processing for all will follow
    cell_ids = np.unique(labels_ph[0])
    fluo_detected = np.unique(added_labels[0][labels_ph[0] == cell_ids[10]])
    num2fluo = get_num2fluo()
    channels = [num2fluo[f] for f in fluo_detected]
    channels = list(np.concatenate(channels).flat)
    channels_unique = np.unique(channels)

    # option: add channel to track csv
    # use cell ID and timeframe to localize cell in df and add detected channels

if __name__ == "__main__":
    file_ph = '../../../midap-datasets/Simon-GlucoseAcetate/GlucoseAcetateExampleData/Data/pos1/ph/track_output/tracking_strack.h5'
    file_gfp = '../../../midap-datasets/Simon-GlucoseAcetate/GlucoseAcetateExampleData/Data/pos1/gfp/track_output/tracking_strack.h5'
    file_mcherry = '../../../midap-datasets/Simon-GlucoseAcetate/GlucoseAcetateExampleData/Data/pos1/mCherry/track_output/tracking_strack.h5'
    main(file_ph, file_gfp, file_mcherry)