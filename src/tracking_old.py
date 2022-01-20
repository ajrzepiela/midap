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
#import utilities as utils


# generate input for unet
def gen_input(img_names_sort, seg_names_sort, cur_frame, target_shape):
    img_cur_frame = resize(io.imread(img_names_sort[cur_frame]), target_shape, order=1)
    img_prev_frame = resize(io.imread(img_names_sort[cur_frame - 1]), target_shape, order=1)
    seg_cur_frame = (resize(io.imread(seg_names_sort[cur_frame]), target_shape, order=0) > 0).astype(int)
    seg_prev_frame = (resize(io.imread(seg_names_sort[cur_frame-1]), target_shape, order=0) > 0).astype(int)

    #inputs_cur_frame = []
    label_prev_frame = label(seg_prev_frame)
    label_cells = np.unique(label_prev_frame)
    num_cells = len(label_cells) - 1

    input_cur_frame = np.empty((target_shape[0], target_shape[1], 4))
    input_cur_frame[:,:,0] = img_prev_frame
    input_cur_frame[:,:,1] = label_prev_frame
    input_cur_frame[:,:,2] = img_cur_frame
    input_cur_frame[:,:,3] = seg_cur_frame
    return input_cur_frame


def track_cell(cell_id, inputs):
    seed = (inputs[:,:,1] == cell_id).astype(int)
    inputs_cell = np.empty(inputs.shape)
    inputs_cell[:,:,[0,2,3]] = inputs[:,:,[0,2,3]]
    inputs_cell[:,:,1] = seed
    results = model.predict(np.array((inputs_cell,)),verbose=0,workers=20,use_multiprocessing=True)

    return results[0,:,:,:]


def track_cur_frame(cur_frame, img_names_sort, seg_names_sort, target_size):
    inputs = gen_input(img_names_sort, seg_names_sort, cur_frame, target_size)
    if inputs.any(): # If not, probably first frame only
        # Predict:
        results_ar = []

        cell_ids = np.unique(inputs[:,:,1])[1:].astype(int)
        #with Pool(processes=20) as pool:
        #    results = pool.map(partial(track_cell, inputs=inputs), cell_ids)
        for i in cell_ids:
            results = track_cell(i, inputs)
            #seed = (inputs[:,:,1] == i).astype(int)
            #inputs_cell = np.empty(inputs.shape)
            #inputs_cell[:,:,[0,2,3]] = inputs[:,:,[0,2,3]]
            #inputs_cell[:,:,1] = seed
            #results = model.predict(np.array((inputs_cell,)),verbose=0)
            results_ar.append(results)

        #print(process.memory_info().rss*1e-9)

    return np.array(results_ar), inputs[:,:,3], inputs



#for cur_frame in tqdm(range(1, num_time_steps)):
#    results_cur_frame, seg_cur_frame, inputs_cur_frame = track_cur_frame(cur_frame, img_names_sort, seg_names_sort, target_size)
#
#    results_all.append(results_cur_frame)
#    inputs_seg.append(seg_cur_frame)
#    inputs_all.append(inputs_cur_frame)


# Save data
#np.savez('inputs_all_red.npz', inputs_all=np.array(inputs_all))
#np.savez('results_all_red.npz', results_all_red=np.array(results_all_red))
#np.savez('label_stack.npz', label_stack=np.array(label_stack))
