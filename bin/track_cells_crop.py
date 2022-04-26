import numpy as np
import glob

import argparse

import sys
sys.path.append('../src')
from tracking import Tracking

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path', help='path to folder for one with specific channel')
parser.add_argument('--start_frame', help='first frame to track')
parser.add_argument('--end_frame', help='last frame to track')
args = parser.parse_args()

# Load data
images_folder = args.path + 'cut_im/'
segmentation_folder = args.path + 'seg_im/'
output_folder = args.path + 'track_output/'
model_file = '../model_weights/model_weights_tracking/unet_moma_track_multisets.hdf5'

img_names_sort = np.sort(glob.glob(images_folder + '*frame*')
                         )[int(args.start_frame):int(args.end_frame)]
seg_names_sort = np.sort(glob.glob(
    segmentation_folder + '*frame*'))[int(args.start_frame):int(args.end_frame)]

# Parameters:
crop_size = (128, 128)
target_size = (512, 512)
input_size = crop_size + (4,)
num_time_steps = len(img_names_sort)

# Process
tr = Tracking(img_names_sort, seg_names_sort, model_file,
              input_size, target_size, crop_size)
tr.track_all_frames_crop()

# Reduce results file for storage
results_all_red = np.zeros(
    (len(tr.results_all), *tr.results_all[0][0, :, :, :2].shape))

for t in range(len(tr.results_all)):
    for ix, cell_id in enumerate(tr.results_all[t]):
        results_all_red[t, cell_id[:, :, 0] > 0.5, 0] = ix+1
        results_all_red[t, cell_id[:, :, 1] > 0.5, 1] = ix+1

# Save data
np.savez(output_folder + 'inputs_all_red.npz',
         inputs_all=np.array(tr.inputs_all))
np.savez(output_folder + 'results_all_red.npz',
         results_all_red=np.array(results_all_red))
