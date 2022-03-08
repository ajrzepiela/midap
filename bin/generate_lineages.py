import numpy as np
import pickle

import sys
sys.path.append('../src/')

from lineage import Lineages


# Load data
data_res = np.load('../data/results_all_red.npz')
results_all_red = data_res['results_all_red']

data_inp = np.load('../data/inputs_all_red.npz')
inputs_all = data_inp['inputs_all']

lin = Lineages(inputs_all, results_all_red)

lin.generate_lineages()

np.savez('../data/label_stack.npz', label_stack=lin.label_stack)
with open('../data/label_dict.pkl', 'wb') as f:
    pickle.dump(lin.label_dict, f)
