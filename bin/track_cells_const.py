import numpy as np
import glob

import argparse

import sys
sys.path.append('../src')
from tracking import Tracking

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path to folder for one with specific channel')
parser.add_argument('--num_frames', help='number of time frames')
args = parser.parse_args()

# Load data
#DeLTA_data = '../Example_For_Tracking-Co/1_5minCo/pos7/GFP/'
images_folder = args.path + 'xy1/phase/'
segmentation_folder = args.path + 'seg_im/'
output_folder = args.path + 'track_output/'
model_file = '../model_weights/unet_moma_track_multisets.hdf5'

img_names_sort = np.sort(glob.glob(images_folder + '*frame*'))[:int(args.num_frames)]
seg_names_sort = np.sort(glob.glob(segmentation_folder + '*frame*'))[:int(args.num_frames)]

# Parameters:
target_size = (512, 512)
input_size = target_size + (1,)
num_time_steps = len(img_names_sort)

# Process
tr = Tracking(img_names_sort, seg_names_sort, model_file, input_size, target_size)
tr.track_all_frames_const()

# Reduce results file for storage
results_all_red = np.empty((len(tr.results_all), *tr.results_all[0][0,:,:,:2].shape))
for t in range(len(tr.results_all)):
    for ix, cell_id in enumerate(tr.results_all[t]):
        results_all_red[t,cell_id[:,:,0] > 0.5, 0] = ix+1
        results_all_red[t,cell_id[:,:,1] > 0.5, 1] = ix+1

# Save data
np.savez(output_folder + 'inputs_all_red_const.npz', inputs_all=np.array(tr.inputs_all))
np.savez(output_folder + 'results_all_red_const.npz', results_all_red=np.array(results_all_red))
