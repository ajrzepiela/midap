import numpy as np
import pickle

import argparse

import sys
sys.path.append('../src/')

from lineage import Lineages

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path to folder with tracking output')
args = parser.parse_args()

# Load data
#data_res = np.load('../data/results_all_red.npz')
#data_res = np.load('../data/results_all_red_crop.npz')
data_res = np.load(args.path + 'results_all_red.npz')
results_all_red = data_res['results_all_red']

#data_inp = np.load('../data/inputs_all_red.npz')
#data_inp = np.load('../data/inputs_all_red_crop.npz')
data_inp = np.load(args.path + 'inputs_all_red.npz')
inputs_all = data_inp['inputs_all']

lin = Lineages(inputs_all, results_all_red)

lin.generate_lineages()

#np.savez('../data/label_stack_test.npz', label_stack=lin.label_stack)
#with open('../data/label_dict_test.pkl', 'wb') as f:
#    pickle.dump(lin.label_dict, f)

#np.savez('../data/label_stack_crop.npz', label_stack=lin.label_stack)
#with open('../data/label_dict_crop.pkl', 'wb') as f:
#    pickle.dump(lin.label_dict, f)

np.savez(args.path + 'label_stack_crop.npz', label_stack=lin.label_stack)
with open(args.path + 'label_dict_crop.pkl', 'wb') as f:
    pickle.dump(lin.label_dict, f)
