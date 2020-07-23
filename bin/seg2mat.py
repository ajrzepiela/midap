import argparse
import os
import numpy as np
from skimage import io
import sys
sys.path.append('../src')

from image_segmentation import save_mat

parser = argparse.ArgumentParser()
parser.add_argument("--path_cut")
parser.add_argument("--path_seg")
parser.add_argument("--path_channel")
args = parser.parse_args()

file_name_cuts = os.listdir(args.path_cut)
file_name_segs = os.listdir(args.path_seg)

cuts = np.array([io.imread(args.path_cut + fc) for fc in file_name_cuts])
segs = np.array([io.imread(args.path_seg + fs) for fs in file_name_segs])

save_mat(cuts, segs, args.path_channel)
