import numpy as np
import pickle
import h5py

import argparse

import sys
sys.path.append('../src/')
from lineage import Lineages


parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path to folder with tracking output')
args = parser.parse_args()

# Load data
data_res = np.load(args.path + 'results_all_red.npz')
results_all_red = data_res['results_all_red']

data_inp = np.load(args.path + 'inputs_all_red.npz')
inputs_all = data_inp['inputs_all']

lin = Lineages(inputs_all, results_all_red)

lin.generate_lineages()

# Save data
lin.track_output.to_csv(args.path + 'track_output.csv', index = True)

hf = h5py.File(args.path + 'track_output.h5', 'w')
hf.create_dataset('inputs_all', data=inputs_all)
hf.create_dataset('results_all', data=results_all_red)
hf.create_dataset('label_stack', data=lin.label_stack)
hf.close()
