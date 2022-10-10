import numpy as np
import pandas as pd

import sys
sys.path.append('../src/')
from lineage import Lineages

def setUp():
    path = '../example_data/Pos57/TXRED/track_output/'

    data_res = np.load(path + 'results_all_red.npz')
    results_all_red = data_res['results_all_red']

    data_inp = np.load(path + 'inputs_all_red.npz')
    inputs_all = data_inp['inputs_all']

    lin = Lineages(inputs_all, results_all_red)
    lin.generate_lineages()

    df = lin.track_output
    df.to_csv('track_output.csv')


def test_global_labels():
    '''Test if sum of local IDs is equal to max of global IDs.
    '''

    track_output = pd.read_csv('track_output.csv', index_col='Unnamed: 0')
    sum_local_ID = sum([track_output[track_output['frame'] == f]
                        ['labelID'].max() for f in pd.unique(track_output['frame'])])
    max_global_ID = track_output.index.max()

    assert sum_local_ID == max_global_ID

def test_ID_assigmment():
    '''Test asssignment of mother and daughter IDs.
    '''

    track_output = pd.read_csv('track_output.csv', index_col='Unnamed: 0')

    ix_split = np.where(track_output.split == 1)[0][0] + 1
    trackID = track_output.loc[ix_split].trackID
    frame = track_output.loc[ix_split].frame
    trackID_d1 = track_output.loc[ix_split].trackID_d1
    
    trackID_mother = track_output.loc[np.where((track_output.frame == frame + 1) & \
                        (track_output.trackID == trackID_d1))[0][0] + 1].trackID_mother

    assert trackID == trackID_mother


# Run setUp of dataframe
setUp()
