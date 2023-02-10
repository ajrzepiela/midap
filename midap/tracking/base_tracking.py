import os
from abc import ABC, abstractmethod
from typing import Collection, Optional, Iterable, Union

import datetime
import numpy as np
import psutil
import skimage.io as io
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.transform import resize
import time
from tqdm import tqdm

from .delta_lineage import DeltaTypeLineages
from ..utils import get_logger

process = psutil.Process(os.getpid())

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = int(os.environ["__VERBOSE"])
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)


class Tracking(ABC):
    """
    A class for cell tracking using the U-Net
    """

    # this logger will be shared by all instances and subclasses
    logger = logger

    def __init__(self, imgs: Collection[str],
                 segs: Optional[Collection[str]] = None,
                 model_weights: Union[str, bytes, os.PathLike, None] = None,
                 input_size: Optional[tuple] = None,
                 target_size: Optional[tuple] = None,
                 crop_size: Optional[tuple] = None,
                 connectivity: Optional[int] = None):
        """
        Initializes the class instance
        :param imgs: List of files containing the cut out images ordered chronological in time
        :param segs: List of files containing the segmentation ordered in the same way as imgs
        :param model_weights: Path to the tracking model weights
        :param input_size: A tuple of ints indicating the shape of the input
        :param target_size: A tuple of ints indicating the shape of the target size
        :param crop_size: A tuple of ints indicating the shape of the crop size
        """

        # set the variables
        self.imgs = imgs
        self.segs = segs

        if self.segs is not None:
            self.num_time_steps = len(self.segs)

        self.model_weights = model_weights
        self.input_size = input_size
        self.target_size = target_size

        if crop_size is not None:
            self.crop_size = crop_size

        if connectivity is not None:
            self.connectivity = connectivity

    def load_data(self, cur_frame: int):
        """
        Loads and resizes raw images and segmentation images of the previous and current time frame.
        :param cur_frame: Number of the current frame.
        :return: The loaded and resized images of the current frame, the previous frame, the current segmentation and
                the previous segmentation
        """

        img_cur_frame = resize(io.imread(self.imgs[cur_frame]), self.target_size, order=1)
        img_prev_frame = resize(io.imread(self.imgs[cur_frame - 1]), self.target_size, order=1)
        seg_cur_frame = (resize(io.imread(self.segs[cur_frame]) > 0, self.target_size, order=0))#.astype(int)
        seg_prev_frame = (resize(io.imread(self.segs[cur_frame - 1]) > 0, self.target_size, order=0))#.astype(int)

        return img_cur_frame, img_prev_frame, seg_cur_frame, seg_prev_frame

    @abstractmethod
    def track_all_frames(self, *args, **kwargs):
        """
        This is an abstract method forcing subclasses to implement it
        """
        pass


class DeltaTypeTracking(Tracking):
    """
    A class for cell tracking using the U-Net Delta V2 model
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DeltaV2Tracking using the base class init
        :param args: Arguments used for the base class init
        :param kwargs: Keyword arguments used for the baseclass init
        """

        # base class init
        super().__init__(*args, **kwargs)

    def track_all_frames(self, output_folder: Union[str, bytes, os.PathLike]):
        """
        Tracks all frames and saves the results to the given output folder
        :param output_folder: The folder to save the results
        """
        # Display estimated runtime
        self.print_process_time()

        # Run tracking
        inputs, results = self.run_model_crop()
        self.store_data(output_folder, inputs, results)

        if results is not None:
            lin = DeltaTypeLineages(inputs=np.array(inputs), results=results)
            lin.store_lineages(output_folder=output_folder)
        else:
            logger.warning("Tracking did not generate any output!")

    def gen_input_crop(self, cur_frame: int):
        """
        Generates the input for the tracking network using cropped images.
        :param cur_frame: Number of the current frame.
        :return: Cropped input for the tracking network
        """

        # Load data
        img_cur_frame, img_prev_frame, seg_cur_frame, seg_prev_frame = self.load_data(cur_frame)

        # Label of the segmentation of the previous frame
        label_prev_frame, num_cells = label(seg_prev_frame, return_num=True, connectivity=self.connectivity)
        label_cur_frame = label(seg_cur_frame, connectivity=self.connectivity)

        props = regionprops(label_prev_frame)

        # create the input
        input_whole_frame = np.stack([img_prev_frame, label_prev_frame, img_cur_frame, seg_cur_frame], axis=-1)

        # Crop images/segmentations per cell and combine all images/segmentations for input
        input_cur_frame = np.zeros((num_cells, self.crop_size[0], self.crop_size[1], 4))
        crop_box = {}
        for cell_ix, p in enumerate(props):
            # get the center
            row, col = p.centroid

            # create the cropbox
            radius_row = self.crop_size[0] / 2
            radius_col = self.crop_size[1] / 2

            # take care of going out of the image
            min_row = np.maximum(0, int(row - radius_row))
            min_col = np.maximum(0, int(col - radius_col))
            max_row = min_row + self.crop_size[0]
            max_col = min_col + self.crop_size[1]

            # take care of overshooting
            if max_row > img_cur_frame.shape[0]:
                max_row = img_cur_frame.shape[0]
                min_row = max_row - self.crop_size[0]
            if max_col > img_cur_frame.shape[1]:
                max_col = img_cur_frame.shape[1]
                min_col = max_col - self.crop_size[1]

            # get the image with just the current label
            seed = (label_prev_frame[min_row:max_row, min_col:max_col] == p.label).astype(int)
            label_cur_frame_crop = label_cur_frame[min_row:max_row, min_col:max_col]
            # remove cells that were split during the crop
            seg_clean = self.clean_crop(label_cur_frame, label_cur_frame_crop)

            cell_ix = p.label - 1
            input_cur_frame[cell_ix, :, :, 0] = img_prev_frame[min_row:max_row, min_col:max_col]
            input_cur_frame[cell_ix, :, :, 1] = seed
            input_cur_frame[cell_ix, :, :, 2] = img_cur_frame[min_row:max_row, min_col:max_col]
            input_cur_frame[cell_ix, :, :, 3] = seg_clean

            crop_box[cell_ix] = (min_row, min_col, max_row, max_col)

        return input_cur_frame, input_whole_frame, crop_box

    def clean_crop(self, seg: np.ndarray, seg_crop: np.ndarray):
        """
        Cleans the cropped segmentation by removing all cells which have been cut during the cropping.
        :param seg: Segmentation of full image.
        :param seg_crop: Segmentation of cropped image.
        :return: The cleaned up segmentation
        """

        # Generate dictionary with cell indices as keys and area as values for the full and cropped segmentation.
        regs = regionprops(seg)
        regs_crop = regionprops(seg_crop)

        areas = {r.label: r.area for r in regs}
        areas_crop = {r.label: r.area for r in regs_crop}

        # Compare area of cell in full and cropped segmentation and remove cells which are smaller than original cell.
        seg_clean = seg_crop.copy()
        for k in areas_crop:
            if areas_crop[k] != areas[k]:
                seg_clean[seg_crop == k] = 0

        seg_clean_bin = (seg_clean > 0).astype(int)

        return seg_clean_bin

    def check_process_time(self):
        """
        Estimates time needed for tracking based on tracking for one frame.
        :return: time in microseconds
        """
        self.logger.info('Estimate needed time for tracking. This may take a while...')
        
        start = time.time()
        self.load_model()
        inputs_cur_frame, input_whole_frame, crop_box = self.gen_input_crop(1)
        results_cur_frame_crop = self.model.predict(inputs_cur_frame, verbose=0)
        end = time.time()
        
        process_time = int((end-start)*1e3)

        return process_time

    def print_process_time(self):
        """
        Prints estimated time for tracking of all frames
        """

        process_time = self.check_process_time()

        print("".join(['\n','─' * 30]))
        print("PLEASE NOTE \nTracking will take: \n ")
        print(" ".join([str(datetime.timedelta(microseconds=process_time)), "hours \n"]))
        print("If the processing time is too \nlong, please consider to cancel \nthe tracking and restart it \non the cluster.")
        print("".join(['─' * 30,'\n']))

    def run_model_crop(self):
        """
        Runs the tracking model
        :return: Arrays containing input and reduced output of Delta model
        """

        # Load model
        self.load_model()

        # Loop over all time frames
        inputs_all = np.empty((self.num_time_steps-1,) + self.target_size + (4,))
        results_all = np.empty((self.num_time_steps-1,) + self.target_size + (2,))

        ram_usg = process.memory_info().rss * 1e-9
        for cur_frame in (pbar := tqdm(range(1, self.num_time_steps), postfix={"RAM": f"{ram_usg:.1f} GB"})):
            inputs_cur_frame, input_whole_frame, crop_box = self.gen_input_crop(cur_frame)

            # check if there is a segmentation
            if inputs_cur_frame.size > 0:
                results_cur_frame_crop = self.model.predict(inputs_cur_frame, verbose=0)
            else:
                results_cur_frame_crop = np.empty_like(inputs_cur_frame)

            # Combine cropped results in one image
            results_cur_frame = np.zeros((results_cur_frame_crop.shape[0], self.target_size[0],
                                          self.target_size[1], 1))

            for i in range(len(crop_box)):
                row_min, col_min, row_max, col_max = crop_box[i]
                results_cur_frame[i, row_min:row_max, col_min:col_max,:] = results_cur_frame_crop[i]

            # Clean results and add to array
            results_cur_frame_clean = self.clean_cur_frame(input_whole_frame[:, :, 3], results_cur_frame)
            results_cur_frame_red = self.reduce_results(results_cur_frame_clean)

            if results_cur_frame_red is not None:
                results_all[cur_frame-1] = results_cur_frame_red

            inputs_all[cur_frame-1] = input_whole_frame

            ram_usg = process.memory_info().rss * 1e-9
            pbar.set_postfix({"RAM": f"{ram_usg:.1f} GB"})

        return inputs_all, results_all

    def clean_cur_frame(self, inp: np.ndarray, res: np.ndarray):
        """
        Clean result from cropped image by comparing the segmentation with the result from the tracking.
        :param inp: Segmentation of full image.
        :param res: Result of the tracking.
        :return: The cleaned up tracking result
        """

        # Labeling of the segmentation.
        inp_label = label(inp, connectivity=self.connectivity)

        # Compare cell from tracking with cell from segmentation and
        # find cells which are overlapping most.
        res_clean = np.zeros(res.shape[:-1] + (2,))
        for ri, r in enumerate(res):

            r_label = label(r[:, :, 0] > 0.9, connectivity=self.connectivity)

            overl = inp_label[np.multiply(inp, r_label) > 0]
            cell_labels = np.unique(overl)
            overl_areas = [np.count_nonzero(overl == l) for l in cell_labels]
            ix_max_overl = np.argsort(overl_areas)[-2:]
            label_max_overl = cell_labels[ix_max_overl]

            for i, c in enumerate(label_max_overl):
                res_clean[ri, :, :, i][inp_label == c] = 1

        res_clean = np.array(res_clean)
        return res_clean


    def reduce_results(self, results: np.ndarray):
        """
        Reduces the amount of output from the delta model from individual images per cell to full images
        :param output_folder: Where to save the output
        :param results: The results
        :return: The reduced data or None if no cells were tracked
        """

        results_red = np.zeros((1,) + self.target_size + (2,))
        
        if len(results) > 0:
            for ix, cell_id in enumerate(results):
                if cell_id[:, :, 0].sum() > 0:
                    results_red[0, cell_id[:, :, 0] > 0, 0] = ix + 1
                if cell_id[:, :, 1].sum() > 0:
                    results_red[0, cell_id[:, :, 1] > 0, 1] = ix + 1

            return results_red

    def store_data(self, output_folder: Union[str, bytes, os.PathLike], input: np.ndarray, result: np.ndarray):
        """ 
        Saves input and output from the Delta model
        :param output_folder: Where to save the output
        :param inputs: The inputs used for the tracking
        :param results: The results
        :return: None
        """

        np.savez(os.path.join(output_folder, 'inputs_all_red.npz'), inputs_all=input)
        np.savez(os.path.join(output_folder, 'results_all_red.npz'), results_all_red=result)

    @abstractmethod
    def load_model(self):
        """
        This is an abstract method forcing subclasses to implement it
        """
        pass
