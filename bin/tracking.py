import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pdb
import sys
sys.path.append('../delta')

from data import predictCompilefromseg_track #saveResult_track, 
from model import unet_track
import utilities as utils
from skimage.measure import label, regionprops
import skimage.io as io
from skimage.transform import resize
import pickle
import h5py
import os, psutil
process = psutil.Process(os.getpid())

import timeit
start = timeit.timeit()

DeLTA_data = '../data/tracking/trial2/pos8/mCherry/'
images_folder = DeLTA_data + 'xy1/phase/'
segmentation_folder = DeLTA_data + 'seg_im/'
outputs_folder = DeLTA_data + 'evaluation/track_output/'
model_file = '../delta/model_weights/unet_moma_track_multisets.hdf5'

# set number of frames
num = 30

img_names_sort = np.sort(os.listdir(images_folder))[:num]
seg_names_sort = np.sort(os.listdir(segmentation_folder))[:num]

# generate input for unet
def gen_input(img_names_sort, seg_names_sort, cur_frame, target_shape):
    img_cur_frame = resize(io.imread(images_folder + img_names_sort[cur_frame]), target_shape, order=1)
    img_prev_frame = resize(io.imread(images_folder + img_names_sort[cur_frame - 1]), target_shape, order=1)
    seg_cur_frame = (resize(io.imread(segmentation_folder + seg_names_sort[cur_frame]), target_shape, order=0) > 0).astype(int)
    seg_prev_frame = (resize(io.imread(segmentation_folder + seg_names_sort[cur_frame-1]), target_shape, order=0) > 0).astype(int)
    
    #inputs_cur_frame = []
    label_prev_frame = label(seg_prev_frame)
    label_cells = np.unique(label_prev_frame)
    num_cells = len(label_cells) - 1
    
    input_cur_frame = np.zeros((num_cells, target_shape[0], target_shape[1], 4))
    for i, l in enumerate(label_cells[1:]):
        seed = (label_prev_frame == l).astype(int)
        input_cur_frame[i,:,:,0] = img_prev_frame
        input_cur_frame[i,:,:,1] = seed
        input_cur_frame[i,:,:,2] = img_cur_frame
        input_cur_frame[i,:,:,3] = seg_cur_frame
    return input_cur_frame

# Parameters:
target_size = (512, 512)
input_size = target_size + (4,)
num_time_steps = len(img_names_sort)

# Load up model:
model = unet_track(input_size = input_size)
model.load_weights(model_file)

# Process
inputs_seg = []
time_points = []

for cur_frame in range(1, num_time_steps):
    inputs = gen_input(img_names_sort, seg_names_sort, cur_frame, target_size)
    print(inputs.shape)
    if inputs.any(): # If not, probably first frame only
        # Predict:
        results_ar = []
        for i in range(inputs.shape[0]):
            results = model.predict(np.array((inputs[i,:,:,:],)),verbose=1)
            results_ar.append(results[0,:,:,:])

        if cur_frame > 1:
            hf = h5py.File('results.h5', 'a')
            hf.create_dataset('result_' + str(cur_frame), data = np.array(results_ar))
            hf.close()
            
            hf = h5py.File('inputs.h5', 'a')
            hf.create_dataset('input_' + str(cur_frame), data = inputs)
            hf.close()

        elif cur_frame == 1:
            
            hf = h5py.File('results.h5', 'w')
            hf.create_dataset('result_' + str(cur_frame), data = np.array(results_ar))
            hf.close()

            hf = h5py.File('inputs.h5', 'w')
            hf.create_dataset('input_' + str(cur_frame), data = np.array(inputs))
            hf.close()
        
    print(process.memory_info().rss*1e-9)

# create input variables for lineage generation
hf_inputs = h5py.File('inputs.h5', 'r')
timepoints=len(hf_inputs.keys()) - 1
chamber_number = 0
num_track_events = [hf_inputs.get(k).shape[0] for k in list(hf_inputs.keys())[:timepoints]] #[len(i) for i in inputs_all[:timepoints]]
frame_numbers = np.concatenate([[i]*num_track_events[i] for i in range(timepoints)])
seg = np.array([np.array(hf_inputs.get(k))[0,:,:,3] for k in list(hf_inputs.keys())[:timepoints]])

# generate lineages
label_stack = np.zeros([timepoints,seg.shape[1],seg.shape[2]],dtype=np.uint16)
lin, label_stack = utils.updatelineage(seg[chamber_number*timepoints], label_stack) # Initialize lineage and label stack on first frame
hf_results = h5py.File('results.h5', 'r')
for i in range(1,timepoints):
    frame_idxs = [x for x, fn in enumerate(frame_numbers) if fn==i]
    if frame_idxs:
        track = np.array(hf_results.get('result_' + str(i+1)))
        track_input = np.array(hf_inputs.get('input_' + str(i+1)))
        scores = utils.getTrackingScores(track_input[0,:,:,3], track)
        attrib = utils.getAttributions(scores)
        print(attrib.shape)
    else:
        attrib = []
    lin, label_stack = utils.updatelineage(seg[chamber_number*timepoints + i], label_stack, framenb=i, lineage=lin, attrib=attrib) # Because we use uint16, we can only track up to 65535 cells per chamber
hf_inputs.close()
hf_results.close()

# save results
np.savez('label_stack.npz', label_stack=label_stack)

end = timeit.timeit()
print(end - start)
