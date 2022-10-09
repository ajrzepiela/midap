import skimage.io as io
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.util import img_as_float
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects

import numpy as np
from tqdm import tqdm

from model_trackingv2 import unet_track

import matplotlib.pyplot as plt

import os
import psutil
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
    crop_size: tuple
        size of cropped input image

    Methods
    -------
    load_data(self, cur_frame)
        Loads and resizes raw images and segmentation images of the previous and current time frame.
    gen_input(cur_frame)
        Generates input for tracking network per time frame.
    gen_input_const(self, cur_frame)
        Generates the input for the tracking network.
    gen_input_crop(self, cur_frame)
        Generates the input for the tracking network using cropped images.
    clean_crop(self, seg, seg_crop)
        Cleans the cropped segmentation by removing all cells which have been cut during the cropping.
    areas2dict(self, regs)
        Generates dictionary based on regionsprops of segmentation.
        The dictionary contains cell indices as keys and the areas as values.
    load_model(self, constant_input=None)
        Loads model for inference/tracking.
    track_cell()
        Tracks single cell within current time frame by using a U-Net.
    track_cur_frame(cur_frame)
        Loops over all cells of current time frame.
    track_all_frames()
        Loops over all time frames to track all cells over time.
    track_all_frames_const(self)
        Track all frames using a constant input.
    track_all_frames_crop(self)
        Track all frames using cropped images as input.
    clean_cur_frame(self, inp, res)
        Clean result from cropped image by comparing the segmentation with the result from the tracking.
    """

    def __init__(self, imgs, segs, model_weights, input_size, target_size, crop_size=None):
        self.imgs = imgs
        self.segs = segs
        self.num_time_steps = len(self.imgs)

        self.model_weights = model_weights
        self.input_size = input_size
        self.target_size = target_size
        if crop_size:
            self.crop_size = crop_size

    def load_data(self, cur_frame):
        """Loads and resizes raw images and segmentation images of the previous and current time frame.

        Parameters
        ----------
        cur_frame: int
            Number of the current frame.
        """

        img_cur_frame = resize(
            io.imread(self.imgs[cur_frame]), self.target_size, order=1)
        img_prev_frame = resize(
            io.imread(self.imgs[cur_frame - 1]), self.target_size, order=1)
        seg_cur_frame = (resize(
            io.imread(self.segs[cur_frame]), self.target_size, order=0) > 0).astype(int)
        seg_prev_frame = (resize(
            io.imread(self.segs[cur_frame-1]), self.target_size, order=0) > 0).astype(int)

        return img_cur_frame, img_prev_frame, seg_cur_frame, seg_prev_frame

    def gen_input(self, cur_frame):
        """Generates the input for the tracking network.

        Parameters
        ----------
        cur_frame: int
            Number of the current frame.
        """

        # Load data
        img_cur_frame, img_prev_frame, seg_cur_frame, seg_prev_frame = self.load_data(
            cur_frame)

        # Label of the segmentation of the previous frame
        label_prev_frame = label(seg_prev_frame)
        len(np.unique(label_prev_frame)) - 1

        # Combine all images and segmentations for input of current frame
        input_cur_frame = np.empty(
            (self.target_size[0], self.target_size[1], 4))
        input_cur_frame[:, :, 0] = img_prev_frame
        input_cur_frame[:, :, 1] = label_prev_frame
        input_cur_frame[:, :, 2] = img_cur_frame
        input_cur_frame[:, :, 3] = seg_cur_frame

        return input_cur_frame

    def gen_input_const(self, cur_frame):
        """Generates the input for the tracking network.

        Parameters
        ----------
        cur_frame: int
            Number of the current frame.
        """

        # Load data
        img_cur_frame, img_prev_frame, seg_cur_frame, seg_prev_frame = self.load_data(
            cur_frame)

        # Label of the segmentation of the previous frame
        label_prev_frame = label(seg_prev_frame)
        label_cells = np.unique(label_prev_frame)
        num_cells = len(label_cells) - 1

        # Combine all images and segmentations for input of current frame
        input_cur_frame = np.empty(
            (num_cells, self.target_size[0], self.target_size[1], 4))

        for n in range(1, num_cells):
            input_cur_frame[n, :, :, 0] = img_prev_frame
            input_cur_frame[n, :, :, 1] = (label_prev_frame == n).astype(int)
            input_cur_frame[n, :, :, 2] = img_cur_frame
            input_cur_frame[n, :, :, 3] = seg_cur_frame

        return np.expand_dims(img_prev_frame, axis=[0, -1]), \
            input_cur_frame[:, :, :, 1], \
            np.expand_dims(img_cur_frame, axis=[0, -1]), \
            np.expand_dims(seg_cur_frame, axis=[0, -1])

    def gen_input_crop(self, cur_frame):
        """Generates the input for the tracking network using cropped images.

        Parameters
        ----------
        cur_frame: int
            Number of the current frame.
        """

        # Load data
        img_cur_frame, img_prev_frame, seg_cur_frame, seg_prev_frame = self.load_data(
            cur_frame)

        # Label of the segmentation of the previous frame
        label_prev_frame = label(seg_prev_frame)
        label_cur_frame = label(seg_cur_frame)
        props = regionprops(label_prev_frame)
        num_cells = len(np.unique(label_prev_frame)) - 1

        input_whole_frame = np.zeros(self.target_size + (4,))
        input_whole_frame[:, :, 0] = img_prev_frame
        input_whole_frame[:, :, 1] = label_prev_frame
        input_whole_frame[:, :, 2] = img_cur_frame
        input_whole_frame[:, :, 3] = seg_cur_frame

        # Crop images/segmentations per cell and combine all images/segmentations for input
        input_cur_frame = np.zeros(
            (num_cells, self.crop_size[0], self.crop_size[1], 4))
        crop_box = {}
        for cell_ix, p in enumerate(props):
            row, col = p.centroid

            radius_row = self.crop_size[0]/2
            radius_col = self.crop_size[1]/2

            min_row = np.max([0, int(row - radius_row)])
            min_col = np.max([0, int(col - radius_col)])

            max_row = min_row + self.crop_size[0]
            max_col = min_col + self.crop_size[1]

            if max_row > img_cur_frame.shape[0]:
                max_row = img_cur_frame.shape[0]
                min_row = max_row - self.crop_size[0]

            if max_col > img_cur_frame.shape[1]:
                max_col = img_cur_frame.shape[1]
                min_col = max_col - self.crop_size[1]

            seed = (label_prev_frame[min_row:max_row,
                    min_col:max_col] == p.label).astype(int)
            label_cur_frame_crop = label_cur_frame[min_row:max_row,
                                                   min_col:max_col]
            seg_clean = self.clean_crop(label_cur_frame, label_cur_frame_crop)

            cell_ix = p.label - 1
            input_cur_frame[cell_ix, :, :,
                            0] = img_prev_frame[min_row:max_row, min_col:max_col]
            input_cur_frame[cell_ix, :, :, 1] = seed
            input_cur_frame[cell_ix, :, :,
                            2] = img_cur_frame[min_row:max_row, min_col:max_col]
            input_cur_frame[cell_ix, :, :, 3] = seg_clean

            crop_box[cell_ix] = (min_row, min_col, max_row, max_col)

        return input_cur_frame, input_whole_frame, crop_box

    def clean_crop(self, seg, seg_crop):
        """Cleans the cropped segmentation by removing all cells which have been cut during the cropping.

        Parameters
        ----------
        seg: array of ints
            Segmentation of full image.

        seg_crop: array of ints
            Segmentation of cropped image.
        """

        # Generate dictionary with cell indices as keys and area as values for the full and cropped segmentation.
        regs = regionprops(seg)
        regs_crop = regionprops(seg_crop)

        areas = self.areas2dict(regs)
        areas_crop = self.areas2dict(regs_crop)

        # Compare area of cell in full and cropped segmentation and remove cells which are smaller than original cell.
        seg_clean = seg_crop.copy()
        for k in areas_crop.keys():
            if areas_crop[k] != areas[k]:
                seg_clean[seg_crop == k] = 0

        seg_clean_bin = (seg_clean > 0).astype(int)

        return seg_clean_bin

    def areas2dict(self, regs):
        """Generates dictionary based on regionsprops of segmentation.
        The dictionary contains cell indices as keys and the areas as values.


        Parameters
        ----------
        regs: list of RegionProperties
            Each item contains labeled cell of segmentation image.
        """

        areas = dict()

        for r in regs:
            areas[r.label] = r.area

        return areas

    def load_model(self, constant_input=None):
        """Loads model for inference/tracking.

        Parameters
        ----------
        constant_input: array, optional
            Array containing the constant input (whole raw image and segmentation image) per time frame.
        """

        #self.model = unet_track(self.input_size, constant_input)
        #self.model.load_weights(self.model_weights)
        
        self.model = unet_track(self.model_weights, self.input_size)
        

    def track_cell(self, cell_id, inputs):
        """Tracks single cell by using the U-Net.

        Parameters
        ----------
        cell_id: int
            ID of seed cell.

        inputs: array
            Generated input.
        """

        # Extract seed cell
        seed = (inputs[:, :, 1] == cell_id).astype(int)

        # Combine seed plus images/segmentations for input
        inputs_cell = np.empty(inputs.shape)
        inputs_cell[:, :, [0, 2, 3]] = inputs[:, :, [0, 2, 3]]
        inputs_cell[:, :, 1] = seed

        # Track cell
        results = self.model.predict(np.array((inputs_cell,)), verbose=0)

        return results[0, :, :, :]

    def track_cur_frame(self, cur_frame):
        """Tracks all cells of current time frame by using the U-Net.

        Parameters
        ----------
        cur_frame: int
            Number of the current frame.
        """

        # Generate input
        inputs = self.gen_input(cur_frame)

        # Loop over single cells in image of current time frame.
        results_ar = []
        cell_ids = np.unique(inputs[:, :, 1])[1:].astype(int)
        for i in cell_ids:
            results = self.track_cell(i, inputs)
            results_ar.append(results)
        results_clean = self.clean_cur_frame(inputs[:, :, 3], results_ar)

        return np.array(results_ar), inputs

    def track_all_frames(self):
        """Track all frames.

        """

        # Load model
        self.load_model()

        # Loop over all frames
        self.inputs_all = []
        self.results_all = []

        for cur_frame in tqdm(range(1, self.num_time_steps)):
            results_cur_frame, inputs_cur_frame = self.track_cur_frame(
                cur_frame)
            self.results_all.append(results_cur_frame)
            self.inputs_all.append(inputs_cur_frame)

            print(process.memory_info().rss*1e-9)

    def track_all_frames_const(self):
        """Track all frames using a constant input.

        """

        # Loop over all time frames
        self.inputs_all = []
        self.results_all = []
        for cur_frame in tqdm(range(1, self.num_time_steps)):
            img_prev_frame, labels, img_cur_frame, seg_cur_frame = self.gen_input_const(
                cur_frame)
            constant_input = [img_prev_frame, img_cur_frame, seg_cur_frame]
            self.load_model(constant_input)
            l = np.expand_dims(labels, axis=-1)
            results_cur_frame = self.model.predict(l, verbose=0, batch_size=1)

            # Extract results
            results_clean = self.clean_cur_frame(
                seg_cur_frame[0, :, :, 0], results_cur_frame)
            self.inputs_all.append(seg_cur_frame)
            self.results_all.append(results_clean)
            print(process.memory_info().rss*1e-9)

    def track_all_frames_crop(self):
        """Track all frames using cropped images as input.

        """

        # Load model
        self.load_model()

        # Loop over all time frames
        self.results_all = []
        self.inputs_all = []

        for cur_frame in tqdm(range(1, self.num_time_steps)):
            inputs_cur_frame, input_whole_frame, crop_box = self.gen_input_crop(
                cur_frame)
            self.results_cur_frame_crop = self.model.predict(
                inputs_cur_frame, verbose=0)

           
            # Combine cropped results in one image
            self.results_cur_frame = np.zeros(
                (self.results_cur_frame_crop.shape[0], self.target_size[0], self.target_size[1], 1))

            
            for i in range(len(crop_box)):
                row_min, col_min, row_max, col_max = crop_box[i]
                self.results_cur_frame[i, row_min:row_max, col_min:col_max,
                                       :] = self.results_cur_frame_crop[i]
            
            # Clean results
            self.results_cur_frame_clean = self.clean_cur_frame(
                input_whole_frame[:, :, 3], self.results_cur_frame)
            self.inputs_all.append(input_whole_frame)
            self.results_all.append(self.results_cur_frame_clean)
            
            print(process.memory_info().rss*1e-9)

    def clean_cur_frame(self, inp, res):
        """Clean result from cropped image by comparing the segmentation with the result from the tracking.

        Parameters
        ----------
        inp: array of ints
            Segmentation of full image.

        res: array of floats
            Result of the tracking.
        """

        # Labeling of the segmentation.
        inp_label = label(inp)

        # Compare cell from tracking with cell from segmentation and
        # find cells which are overlapping most.
        res_clean = np.zeros(res.shape[:-1] + (2,))
        for ri, r in enumerate(res):

            r_label = label(r[:,:,0] > 0.9)
            r_label = remove_small_objects(r_label,min_size=5)

            overl = inp_label[np.multiply(inp, r_label) > 0]
            cell_labels = np.unique(overl)
            overl_areas = [np.count_nonzero(overl == l) for l in cell_labels]
            ix_max_overl = np.argsort(overl_areas)[-2:]
            label_max_overl = cell_labels[ix_max_overl]

            for i, c in enumerate(label_max_overl):
                res_clean[ri,:,:,i][inp_label == c] = 1

        res_clean = np.array(res_clean)
        return res_clean


