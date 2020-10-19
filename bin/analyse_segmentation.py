from skimage import io
from skimage.measure import regionprops

import os
import numpy as np

from tqdm import tqdm

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_seg")
parser.add_argument("--path_result")
args = parser.parse_args()


def count_cells(seg):
    return (len(np.unique(seg)) - 1)


def count_killed(seg):
    # compute regionprops
    regions = regionprops(seg)

    # compute ratio between minor and major axis
    minor_to_major = np.array(
        [r.minor_axis_length / r.major_axis_length for r in regions])

    # get number of cells with ratio > 0.5
    num_killed = len(np.where(minor_to_major > 0.7)[0])

    return num_killed


# computer number of living and killed cells
num_cells = []
num_killed = []

for p in tqdm(np.sort(os.listdir(args.path_seg))):
    num_cells.append(count_cells(io.imread(args.path_seg + p)))
    num_killed.append(count_killed(io.imread(args.path_seg + p)))

num_cells = np.array(num_cells)
num_killed = np.array(num_killed)
num_living = num_cells - num_killed
d = {
    'all cells': num_cells,
    'living cells': num_living,
    'killed cells': num_killed}
df_cells = pd.DataFrame(data=d)

# df_cells.loc['all cells'] = num_cells
# df_cells.loc['living cells'] = (num_cells - num_killed)
# df_cells.loc['killed cells'] = num_killed

df_cells.to_csv(args.path_result + 'cell_number.csv')
