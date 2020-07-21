import argparse
import os
import numpy as np

import sys
sys.path.append('../src')
from image_cutout import CutoutImage

parser = argparse.ArgumentParser()
parser.add_argument("--path_ch0")
parser.add_argument("--path_ch1")
parser.add_argument("--path_ch2")
args = parser.parse_args()

files_ch0 = args.path_ch0 + '/' + np.sort(os.listdir(args.path_ch0)).astype(list)
files_ch1 = args.path_ch1 + '/' + np.sort(os.listdir(args.path_ch1)).astype(list)
files_ch2 = args.path_ch2 + '/' + np.sort(os.listdir(args.path_ch2)).astype(list)

cut = CutoutImage(files_ch0, min_x_range = 700, max_x_range = 1600, min_y_range = 500, max_y_range = 1600,
                  ch1 = files_ch1, ch2 = files_ch2)
cut.run_align_cutout()
