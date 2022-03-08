import numpy as np
import glob
from tqdm import tqdm

from functools import partial
from multiprocessing import Pool

from skimage.measure import label, regionprops
import skimage.io as io
from skimage.transform import resize

import os, psutil
process = psutil.Process(os.getpid())

import sys
sys.path.append('../delta')
from model import unet_track
import utilities as utils

import sys
sys.path.append('../src')
from tracking import Tracking

# Load data
DeLTA_data = '../Example_For_Tracking-Co/1_5minCo/pos7/GFP/'#'../../../ackermann-bacteria-segmentation/data/tracking/trial2/pos8/mCherry/'
images_folder = DeLTA_data + 'xy1/phase/'
segmentation_folder = DeLTA_data + 'seg_im/'
outputs_folder = DeLTA_data + 'evaluation/track_output/'
model_file = '../delta/model_weights/unet_moma_track_multisets.hdf5'

num = 10 # set number of frames

img_names_sort = np.sort(glob.glob(images_folder + '*frame*'))[:num]
seg_names_sort = np.sort(glob.glob(segmentation_folder + '*frame*'))[:num]


# Parameters:
target_size = (192,192)#(256, 256)
input_size = target_size + (4,)
num_time_steps = len(img_names_sort)

# Process
tr = Tracking(img_names_sort, seg_names_sort, model_file, input_size, target_size)
tr.track_all_frames_crop()
#tr.track_all_frames()

# Reduce results file for storage
results_all_red = np.zeros((len(tr.results_all), *tr.results_all[0][0,:,:,:2].shape))
for t in range(len(tr.results_all)):
    for ix, cell_id in enumerate(tr.results_all[t]):
        results_all_red[t,cell_id[:,:,0] > 0.5, 0] = ix+1
        results_all_red[t,cell_id[:,:,1] > 0.5, 1] = ix+1

## Save data
#np.savez('../data/inputs_all_red_test.npz', inputs_all=np.array(tr.inputs_all))
np.savez('../data/results_all_red_crop.npz', results_all_red=np.array(results_all_red))
