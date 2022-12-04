from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

import numpy as np
from tqdm import tqdm
from typing import Iterable, Optional

import os
import psutil
process = psutil.Process(os.getpid())

from ..networks.deltav2 import unet_track
from .base_tracking import Tracking
from midap.utils import get_logger

class DeltaV2Tracking(Tracking):
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


    def gen_input_crop(self, cur_frame: int):
        """
        Generates the input for the tracking network using cropped images.
        :param cur_frame: Number of the current frame.
        :return: Cropped input for the tracking network
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

        areas = self.areas2dict(regs)
        areas_crop = self.areas2dict(regs_crop)

        # Compare area of cell in full and cropped segmentation and remove cells which are smaller than original cell.
        seg_clean = seg_crop.copy()
        for k in areas_crop.keys():
            if areas_crop[k] != areas[k]:
                seg_clean[seg_crop == k] = 0

        seg_clean_bin = (seg_clean > 0).astype(int)

        return seg_clean_bin


    def areas2dict(self, regs: Iterable):
        """
        Generates dictionary based on regionsprops of segmentation. The dictionary contains cell indices as keys and
        the areas as values.
        :param regs: Each item contains labeled cell of segmentation image.
        :return: Dictionary of areas
        """

        areas = dict()

        for r in regs:
            areas[r.label] = r.area

        return areas


    def run_tracking(self):
        """
        Loads model for inference/tracking.
        """

        self.model = unet_track(self.model_weights, self.input_size)

        # Load model
        #self.load_model()

        # Loop over all time frames
        self.results_all = []
        self.inputs_all = []

        ram_usg = process.memory_info().rss * 1e-9
        for cur_frame in (pbar := tqdm(range(1, self.num_time_steps), postfix={"RAM": f"{ram_usg:.1f} GB"})):
            inputs_cur_frame, input_whole_frame, crop_box = self.gen_input_crop(
                cur_frame)

            # check if there is a segmentation
            if inputs_cur_frame.size > 0:
                self.results_cur_frame_crop = self.model.predict(
                    inputs_cur_frame, verbose=0)
            else:
                self.results_cur_frame_crop = np.empty_like(inputs_cur_frame)
           
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
            
            ram_usg = process.memory_info().rss*1e-9
            pbar.set_postfix({"RAM": f"{ram_usg:.1f} GB"})


    def clean_cur_frame(self, inp: np.ndarray, res: np.ndarray):
        """
        Clean result from cropped image by comparing the segmentation with the result from the tracking.
        :param inp: Segmentation of full image.
        :param res: Result of the tracking.
        :return: The cleaned up tracking result
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


    def store_data(self, logger, output_folder: str):
        # Reduce results file for storage if there is a tracking result
        if sum([res.size for res in self.results_all]) > 0:
            logger.info("Saving results of tracking...")
            # the first might be emtpy
            results_all_red = np.zeros(
                (len(self.results_all), ) + self.results_all[0].shape[1:3] + (2,))

            for t in range(len(self.results_all)):
                for ix, cell_id in enumerate(self.results_all[t]):
                    if cell_id[:, :, 0].sum() > 0:
                        results_all_red[t, cell_id[:, :, 0] > 0, 0] = ix+1
                    if cell_id[:, :, 1].sum() > 0:
                        results_all_red[t, cell_id[:, :, 1] > 0, 1] = ix+1

            # Save data
            np.savez(os.path.join(output_folder, 'inputs_all_red.npz'),
                    inputs_all=np.array(self.inputs_all))
            np.savez(os.path.join(output_folder, 'results_all_red.npz'),
                    results_all_red=np.array(results_all_red))
