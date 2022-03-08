import skimage.io as io
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.util import img_as_float
from skimage.segmentation import clear_border

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


    def gen_input_crop(self, cur_frame):
        img_cur_frame = io.imread(self.imgs[cur_frame])
        img_prev_frame = io.imread(self.imgs[cur_frame - 1])
        seg_cur_frame = io.imread(self.segs[cur_frame])
        seg_prev_frame = io.imread(self.segs[cur_frame-1])
        
        new_target = (512,512)#img_cur_frame.shape
        img_cur_frame = resize(img_cur_frame, new_target, order=1)
        img_prev_frame = resize(img_prev_frame, new_target, order=1)
        seg_cur_frame = (resize(seg_cur_frame, new_target, order=0) > 0).astype(int)
        seg_prev_frame = (resize(seg_prev_frame, new_target, order=0) > 0).astype(int)

        print(img_cur_frame.shape)
        print(seg_cur_frame.shape)
        label_prev_frame = label(seg_prev_frame)
        props = regionprops(label_prev_frame)
        num_cells = len(np.unique(label_prev_frame)) - 1
        #num_cells
        input_cur_frame = np.zeros((num_cells, self.target_size[0], self.target_size[1], 4))
        crop_box = {}
        for cell_ix, p in enumerate(props):
            print(cell_ix)
            row, col = p.centroid

            radius_row = self.target_size[0]/2
            radius_col = self.target_size[1]/2

            min_row = np.max([0, int(row - radius_row)])
            min_col = np.max([0, int(col - radius_col)])
        
            max_row = min_row + self.target_size[0]
            max_col = min_col + self.target_size[1]

            if max_row > img_cur_frame.shape[0]:
                max_row = img_cur_frame.shape[0]
                min_row = max_row - self.target_size[0]

            if max_col > img_cur_frame.shape[1]:
                max_col = img_cur_frame.shape[1]
                min_col = max_col - self.target_size[1]
            
            seed = (label_prev_frame[min_row:max_row, min_col:max_col] == p.label).astype(int)
            seg_clear = clear_border(seg_cur_frame[min_row:max_row, min_col:max_col])
            input_cur_frame[cell_ix,:,:,0] = img_prev_frame[min_row:max_row, min_col:max_col]
            input_cur_frame[cell_ix,:,:,1] = seed#(label_prev_frame[min_row:max_row, min_col:max_col] == p.label).astype(int)
            input_cur_frame[cell_ix,:,:,2] = img_cur_frame[min_row:max_row, min_col:max_col]
            input_cur_frame[cell_ix,:,:,3] = seg_clear#clear_border(seg_cur_frame[min_row:max_row, min_col:max_col])
            
            
            crop_box[cell_ix] = (min_row, min_col, max_row, max_col)
            
        return input_cur_frame, crop_box


    def gen_input(self, cur_frame):
        img_cur_frame = resize(io.imread(self.imgs[cur_frame]), self.target_size, order=1)
        img_prev_frame = resize(io.imread(self.imgs[cur_frame - 1]), self.target_size, order=1)
        seg_cur_frame = (resize(io.imread(self.segs[cur_frame]), self.target_size, order=0) > 0).astype(int)
        seg_prev_frame = (resize(io.imread(self.segs[cur_frame-1]), self.target_size, order=0) > 0).astype(int)

        label_prev_frame = label(seg_prev_frame)
        #props = regionprops(label_prev_frame)
        num_cells = len(np.unique(label_prev_frame)) - 1

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


    def load_model(self, constant_input=None):
        self.model = unet_track(self.input_size, constant_input)
        #self.model.load_weights(self.model_weights, by_name=True)
        self.model.load_weights(self.model_weights)


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


    def track_all_frames_crop(self):
        self.load_model()

        self.results_all = []

        for cur_frame in tqdm(range(1, self.num_time_steps)):

            self.inputs_cur_frame, self.crop_box = self.gen_input_crop(cur_frame)
            self.results_cur_frame_crop = self.model.predict(self.inputs_cur_frame, verbose=1)
            #self.results_cur_frame_crop = self.model.predict(np.array((self.inputs_cur_frame[0],)), verbose=1)
        
            img = io.imread(self.imgs[cur_frame])
            self.results_cur_frame = np.zeros((self.results_cur_frame_crop.shape[0], 512, 512, 3))
            #self.results_cur_frame = np.zeros((self.results_cur_frame_crop.shape[0], img.shape[0], img.shape[1], 3))

            for i in range(len(self.crop_box)):
                row_min, col_min, row_max, col_max = self.crop_box[i]
                self.results_cur_frame[i,row_min:row_max,col_min:col_max,:] = self.results_cur_frame_crop[i]

            self.results_all.append(self.results_cur_frame)
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
