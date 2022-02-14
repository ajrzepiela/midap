import skimage.io as io
from skimage.measure import label
from skimage.transform import resize

import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../delta')
from model import unet_track

import os, psutil
process = psutil.Process(os.getpid())

class Tracking():
    """
    A class for cell tracking using the U-Net

    ...

    Attributes
    ----------
    imgs: list of str
        list of path strings
    segs : list of str
        list of path strings
    model_weights : str 
        path to model weights
    input_size : tuple
        input size of tracking network
    target_size: tuple
        target size of tracking network

    Methods
    -------
    gen_input(cur_frame)
        Generates input for trackinng network per time frame
    track_cur_frame(cur_frame)
        Loops over all cells of current time frame
    load_model()
        Loads tracking model
    track_cell()
        Tracks single cell within current time frame by using a U-Net
    track_all_frames()
        Loops over all time frames to track all cells over time
    """

    def __init__(self, imgs, segs, model_weights, input_size, target_size):
        self.imgs = imgs
        self.segs = segs
        self.num_time_steps = len(self.imgs)

        self.model_weights = model_weights
        self.input_size = input_size
        self.target_size = target_size
        

        #self.load_model()

    def gen_input_2(self, cur_frame):

        img_cur_frame = resize(io.imread(self.imgs[cur_frame]), self.target_size, order=1)
        img_prev_frame = resize(io.imread(self.imgs[cur_frame - 1]), self.target_size, order=1)
        seg_cur_frame = (resize(io.imread(self.segs[cur_frame]), self.target_size, order=0) > 0).astype(int)
        seg_prev_frame = (resize(io.imread(self.segs[cur_frame-1]), self.target_size, order=0) > 0).astype(int)

        label_prev_frame = label(seg_prev_frame)
        label_cells = np.unique(label_prev_frame)
        num_cells = len(label_cells) - 1

        input_cur_frame = np.empty((num_cells, self.target_size[0], self.target_size[1], 4))
       
        for n in range(1,num_cells):#num_cells
            input_cur_frame[n,:,:,0] = img_prev_frame
            input_cur_frame[n,:,:,1] = (label_prev_frame == n).astype(int)
            input_cur_frame[n,:,:,2] = img_cur_frame
            input_cur_frame[n,:,:,3] = seg_cur_frame
       
        return np.expand_dims(img_prev_frame, axis=[0,-1]), \
               input_cur_frame[:,:,:,1], \
               np.expand_dims(img_cur_frame, axis=[0,-1]), \
               np.expand_dims(seg_cur_frame, axis=[0,-1])
        #return input_cur_frame


    def gen_input(self, cur_frame):
        img_cur_frame = resize(io.imread(self.imgs[cur_frame]), self.target_size, order=1)
        img_prev_frame = resize(io.imread(self.imgs[cur_frame - 1]), self.target_size, order=1)
        seg_cur_frame = (resize(io.imread(self.segs[cur_frame]), self.target_size, order=0) > 0).astype(int)
        seg_prev_frame = (resize(io.imread(self.segs[cur_frame-1]), self.target_size, order=0) > 0).astype(int)

        label_prev_frame = label(seg_prev_frame)
        label_cells = np.unique(label_prev_frame)
        num_cells = len(label_cells) - 1

        input_cur_frame = np.empty((self.target_size[0], self.target_size[1], 4))
        input_cur_frame[:,:,0] = img_prev_frame
        input_cur_frame[:,:,1] = label_prev_frame
        input_cur_frame[:,:,2] = img_cur_frame
        input_cur_frame[:,:,3] = seg_cur_frame

        return input_cur_frame


    def track_cur_frame(self, cur_frame):

        inputs = self.gen_input(cur_frame)
#        if inputs.any(): # If not, probably first frame only
#            # Predict:
        results_ar = []

        cell_ids = np.unique(inputs[:,:,1])[1:].astype(int)
        for i in cell_ids:
            results = self.track_cell(i, inputs)
            results_ar.append(results)

        return np.array(results_ar), inputs


    def load_model(self, constant_input):
        self.model = unet_track(self.input_size, constant_input)
        self.model.load_weights(self.model_weights, by_name=True)


    def track_cell(self, cell_id, inputs):

        seed = (inputs[:,:,1] == cell_id).astype(int)
        inputs_cell = np.empty(inputs.shape)
        inputs_cell[:,:,[0,2,3]] = inputs[:,:,[0,2,3]]
        inputs_cell[:,:,1] = seed
        results = self.model.predict(np.array((inputs_cell,)),verbose=0)
        return results[0,:,:,:]

    def track_all_frames_2(self):
        self.inputs_all = []
        self.results_all = []
        for cur_frame in tqdm(range(1, self.num_time_steps)):
            img_prev_frame, labels, img_cur_frame, seg_cur_frame = self.gen_input_2(cur_frame)
        #print(img_prev_frame.shape)
        #print(labels.shape)
        #print(img_cur_frame.shape)
        #print(seg_cur_frame.shape)


            constant_input = [img_prev_frame, img_cur_frame, seg_cur_frame]
            self.load_model(constant_input)
            l = np.expand_dims(labels,axis=-1)
        #print(l.shape)
        #l2 = np.expand_dims(labels[0,:,:],axis=[0,-1])
            results_cur_frame = self.model.predict(l, verbose=0, batch_size=1)
            #self.results_all.append(results_cur_frame)
            print(results_cur_frame.shape)
            print(process.memory_info().rss*1e-9)


    def track_all_frames(self):
        self.load_model()

        self.inputs_all = []
        self.results_all = []

        for cur_frame in tqdm(range(1, self.num_time_steps)):
            results_cur_frame, inputs_cur_frame = self.track_cur_frame(cur_frame)
            print(inputs_cur_frame.shape)
            self.results_all.append(results_cur_frame)
            #inputs_seg.append(seg_cur_frame)
            self.inputs_all.append(inputs_cur_frame)

        print(process.memory_info().rss*1e-9)
