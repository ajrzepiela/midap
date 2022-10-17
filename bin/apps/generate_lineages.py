import os.path

import numpy as np
import h5py

import argparse
import pandas as pd

from midap.tracking.lineage import Lineages
from midap.utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='path to folder with tracking output')
parser.add_argument("--loglevel", type=int, default=7, help="Loglevel of the script.")
args = parser.parse_args()

# logging
logger = get_logger(__file__, args.loglevel)
logger.info(f"Creating lineages for: {args.path}")

# try to read the data
try:
    # read the results, this will trigger a file not found error if the tracking did not produce anything
    data_res = np.load(os.path.join(args.path, 'results_all_red.npz'))
    results_all_red = data_res['results_all_red']

    # read the inputs
    data_inp = np.load(os.path.join(args.path, 'inputs_all_red.npz'))
    inputs_all = data_inp['inputs_all']

    # generate the lineages
    lin = Lineages(inputs_all, results_all_red)

    lin.generate_lineages()

    # Save data
    lin.track_output.to_csv(args.path + 'track_output.csv', index=True)

    hf = h5py.File(args.path + 'raw_inputs.h5', 'w')
    raw_inputs = inputs_all[:, :, :, 0]
    hf.create_dataset('raw_inputs', data=raw_inputs)
    hf.close()

    hf = h5py.File(args.path + 'segmentations.h5', 'w')
    segs = inputs_all[0, :, :, 3]
    hf.create_dataset('segmentations', data=segs)
    hf.close()

    hf = h5py.File(args.path + 'label_stack.h5', 'w')
    hf.create_dataset('label_stack', data=lin.label_stack)
    hf.close()

except FileNotFoundError:
    # the is no output for the tracking
    logger.warning("The tracking does not have produced any output.")

    # we dump an empty file to show that the code did actually run
    columns = ['frame', 'labelID', 'trackID', 'lineageID', 'trackID_d1', 'trackID_d2', 'split',
               'trackID_mother', 'area', 'edges_min_row', 'edges_min_col', 'edges_max_row',
               'edges_max_col', 'intensity_max', 'intensity_mean', 'intensity_min',
               'minor_axis_length', 'major_axis_length', 'frames',
               'first_frame', 'last_frame']
    track_output = pd.DataFrame(columns=columns)
    track_output.to_csv(os.path.join(args.path, 'track_output.csv'), index=True)
