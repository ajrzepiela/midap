import datetime
import os
import time
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Optional

import numpy as np
import psutil
import skimage.io as io
from scipy.spatial import distance_matrix
from skimage.measure import label, regionprops
from skimage.transform import resize
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

    def __init__(self,
                 imgs: List[Union[str, bytes, os.PathLike]],
                 segs: List[Union[str, bytes, os.PathLike]],
                 model_weights: Optional[Union[str, bytes, os.PathLike]],
                 input_size: Optional[Tuple[int, int, int]] = None,
                 target_size: Optional[Tuple[int, int]] = None,
                 connectivity=1):
        """
        Initializes the class instance
        :param imgs: List of files containing the cut out images ordered chronological in time
        :param segs: List of files containing the segmentation ordered in the same way as imgs
        :param model_weights: Path to the tracking model weights
        :param input_size: A tuple of ints indicating the shape of the input for the networks,
                           this will be increased if necessary
        :param target_size: A tuple of ints indicating the shape of the target size of the input images, if None
                            the images will not be resized after reading
        """

        # set the variables
        self.imgs = imgs
        self.segs = segs
        self.num_time_steps = len(self.segs)
        self.model_weights = model_weights
        if input_size is None:
            self.input_size = (32, 32, 4)
        else:
            self.input_size = input_size
        self.max_input_size = 256
        self.target_size = target_size
        self.connectivity = connectivity

    def load_data(self, cur_frame: int, label=False):
        """
        Loads and resizes raw images and segmentation images of the previous and current time frame.
        :param cur_frame: Number of the current frame.
        :param label: If True, the labelled image is returned, note the binary segmentation
        :return: The loaded and resized images of the current frame, the previous frame, the current segmentation and
                the previous segmentation
        """

        img = io.imread(self.imgs[cur_frame])
        if self.target_size is None:
            target_size = img.shape
        else:
            target_size = self.target_size
        img_cur_frame = resize(img, target_size, order=1)
        img_prev_frame = resize(io.imread(self.imgs[cur_frame - 1]), target_size, order=1)
        if label:
            seg_cur_frame = (resize(io.imread(self.segs[cur_frame]), target_size, order=0))
            seg_prev_frame = (resize(io.imread(self.segs[cur_frame - 1]), target_size, order=0))
        else:
            seg_cur_frame = (resize(io.imread(self.segs[cur_frame]) > 0, target_size, order=0))
            seg_prev_frame = (resize(io.imread(self.segs[cur_frame - 1]) > 0, target_size, order=0))

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
            lin = DeltaTypeLineages(inputs=np.array(inputs), results=results, connectivity=self.connectivity)
            data_file, csv_file = lin.store_lineages(output_folder=output_folder)
        else:
            logger.warning("Tracking did not generate any output!")
            data_file, csv_file = None, None

        return data_file, csv_file

    def gen_input_crop(self, cur_frame: int):
        """
        Generates the input for the tracking network using cropped images.
        :param cur_frame: Number of the current frame.
        :return: Cropped input for the tracking network
        """

        # Load data
        img_cur_frame, img_prev_frame, seg_cur_frame, seg_prev_frame = self.load_data(cur_frame, label=False)

        # Label of the segmentation of the previous frame
        label_prev_frame, num_cells = label(seg_prev_frame, return_num=True, connectivity=self.connectivity)
        label_cur_frame = label(seg_cur_frame, connectivity=self.connectivity)

        # get the props and create an area dict for the current frame
        props_prev = regionprops(label_prev_frame)
        props_curr = regionprops(label_cur_frame)
        areas = {r.label: r.area for r in props_curr}

        # get the distance matrix of the centroids
        centers_prev = np.array([p.centroid for p in props_prev])
        centers_curr = np.array([p.centroid for p in props_curr])
        dist_mat = distance_matrix(centers_prev, centers_curr)

        # if min distance between to cells in the frames is smaller than our input, we adjust to the next higher
        min_dist = int(np.max(np.min(dist_mat, axis=1)))
        # the square crop region should be large enough to fit the biggest cell
        min_dist = np.maximum(min_dist, np.max([r.axis_major_length for r in props_curr]).astype(int))
        # it should not be bigger than the frame itself or max input shape
        min_dist = np.minimum(np.minimum(self.max_input_size, np.min(label_cur_frame.shape)), min_dist)
        if min_dist >= self.input_size[0]:
            self.logger.info(f"Current max dist between cells: {min_dist}, increasing input size of model...")
            self.input_size = (min_dist // 32 + 1) * 32, (min_dist // 32 + 1) * 32, 4
            self.load_model()

        # create the input
        input_whole_frame = np.stack([img_prev_frame, label_prev_frame, img_cur_frame, seg_cur_frame], axis=-1)

        # Crop images/segmentations per cell and combine all images/segmentations for input
        input_cur_frame = np.zeros((num_cells, self.input_size[0], self.input_size[1], 4))
        crop_box = np.zeros((num_cells, 4), dtype=int)
        for cell_ix, p in enumerate(props_prev):
            # get the center
            row, col = p.centroid

            # create the cropbox
            radius_row = self.input_size[0] / 2
            radius_col = self.input_size[1] / 2

            # take care of going out of the image
            min_row = np.maximum(0, int(row - radius_row))
            min_col = np.maximum(0, int(col - radius_col))
            max_row = min_row + self.input_size[0]
            max_col = min_col + self.input_size[1]

            # take care of overshooting
            if max_row > img_cur_frame.shape[0]:
                max_row = img_cur_frame.shape[0]
                min_row = max_row - self.input_size[0]
            if max_col > img_cur_frame.shape[1]:
                max_col = img_cur_frame.shape[1]
                min_col = max_col - self.input_size[1]

            # get the image with just the current label
            seed = (label_prev_frame[min_row:max_row, min_col:max_col] == p.label).astype(int)
            label_cur_frame_crop = label_cur_frame[min_row:max_row, min_col:max_col]
            # remove cells that were split during the crop
            seg_clean = self.clean_crop(areas, label_cur_frame_crop)
            
            cell_ix = p.label - 1
            input_cur_frame[cell_ix, :, :, 0] = img_prev_frame[min_row:max_row, min_col:max_col]
            input_cur_frame[cell_ix, :, :, 1] = seed
            input_cur_frame[cell_ix, :, :, 2] = img_cur_frame[min_row:max_row, min_col:max_col]
            input_cur_frame[cell_ix, :, :, 3] = seg_clean

            crop_box[cell_ix] = min_row, min_col, max_row, max_col

        return input_cur_frame, input_whole_frame, crop_box

    def clean_crop(self, areas: dict, seg_crop: np.ndarray):
        """
        Cleans the cropped segmentation by removing all cells which have been cut during the cropping.
        :param areas: A dict of label -> area of the full segmentation frame
        :param seg_crop: Segmentation of cropped image.
        :return: The cleaned up segmentation
        """

        # FIXME: This function is still fairly inefficient, the best way would be to check the intersection of the
        # FIXME: bounding boxes with the crop, but since this is still much faster than the network pass I am
        # FIXME: currently too lazy to implement it

        # Generate dictionary with cell indices as keys and area as values for the full and cropped segmentation.
        regs_crop = regionprops(seg_crop)

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
        :return: time in milliseconds
        """
        self.logger.info('Estimate needed time for tracking. This may take a while...')
        
        start = time.time()
        self.load_model()
        inputs_cur_frame, input_whole_frame, crop_box = self.gen_input_crop(1)
        _ = self.model.predict(inputs_cur_frame, verbose=0)
        end = time.time()
        
        process_time = int((end-start)*1e3)

        return process_time

    def print_process_time(self):
        """
        Prints estimated time for tracking of all frames
        """

        process_time = self.check_process_time()

        print('\n' + '─' * 30)
        print("PLEASE NOTE \nTracking will take: \n ")
        print(f"{str(datetime.timedelta(milliseconds=process_time))} hours \n")
        print("If the processing time is too \n"
              "long, please consider to cancel \n"
              "the tracking and restart it \n"
              "on the cluster.")
        print("─" * 30 + '\n')

    def run_model_crop(self):
        """
        Runs the tracking model
        :return: Arrays containing input and reduced output of Delta model
        """

        # Load model
        self.load_model()

        # Loop over all time frames
        inputs_all = []
        results_all = []

        ram_usg = process.memory_info().rss * 1e-9
        for cur_frame in (pbar := tqdm(range(1, self.num_time_steps), postfix={"RAM": f"{ram_usg:.1f} GB"})):
            inputs_cur_frame, input_whole_frame, crop_box = self.gen_input_crop(cur_frame)

            # check if there is a segmentation
            if inputs_cur_frame.size > 0:
                results_cur_frame_crop = self.model.predict(inputs_cur_frame.astype(np.float32),
                                                            verbose=0,
                                                            batch_size=128)
            else:
                results_cur_frame_crop = np.empty_like(inputs_cur_frame)

            # Combine cropped results in one image
            results_cur_frame = self.transfer_results(full_shape=input_whole_frame.shape[:2] + (2,),
                                                      inp=inputs_cur_frame,
                                                      res=results_cur_frame_crop,
                                                      crop_boxes=crop_box)

            # add to results
            results_all.append(results_cur_frame)
            inputs_all.append(input_whole_frame)

            ram_usg = process.memory_info().rss * 1e-9
            pbar.set_postfix({"RAM": f"{ram_usg:.1f} GB"})

        return np.array(inputs_all), np.array(results_all)

    def transfer_results(self, full_shape: Tuple[int, int, int], inp: np.ndarray, res: np.ndarray,
                         crop_boxes: np.ndarray):
        """
        Transfers the results to a single frame
        :param full_shape: The full shape of the final image
        :param inp: A stack of cropped images (BWHC) that contain the input of the network
        :param res: The output of the network
        :param crop_boxes: The crop boxes for each input
        :return: An array that is delta v1 like, i.e. WH2 where the first channels dim and second channel dim contain
                 the daughter cells
        """
        target = np.zeros(full_shape)

        for cell_id, (i, r, c) in enumerate(zip(inp, res, crop_boxes)):
            # extract the crop boxes
            row_min, col_min, row_max, col_max = c

            # we get the view of the target that is relevant
            crop_target = target[row_min:row_max, col_min:col_max, :]

            # the relevant images
            i_candidates = i[..., 3]

            # label the candiates
            inp_label = label(i_candidates, connectivity=self.connectivity)
            # get the count of the max overlay
            bin_count = np.bincount(inp_label[r[:, :, 0] > 0.5])
            # this gets the indexes of the two largest count (not including the 0 bin) the largest count is first
            label_max_overl = np.argsort(bin_count[1:])[-1:-3:-1] + 1

            # we need to check if any of the candidates has already been marked
            masks = []
            for num, (color, count) in enumerate(zip(label_max_overl, bin_count[label_max_overl])):
                # we want to have at least 20% overlay to accept the candidate
                if count > 0 and np.sum(mask := inp_label == color)/count > 0.2:
                    if np.all(crop_target[mask, :] == 0):
                        masks.append(mask)

            # assign the succesfull masks, we make a second loop to make sure that one cell is always in the first frame
            for num, mask in enumerate(masks):
                crop_target[..., num][mask] = cell_id + 1

        return target

    def store_data(self, output_folder: Union[str, bytes, os.PathLike], input: np.ndarray, result: np.ndarray):
        """ 
        Saves input and output from the Delta model
        :param output_folder: Where to save the output
        :param inputs: The inputs used for the tracking
        :param results: The results
        """

        np.savez(os.path.join(output_folder, 'inputs_all_red.npz'), inputs_all=input)
        np.savez(os.path.join(output_folder, 'results_all_red.npz'), results_all_red=result)

    @abstractmethod
    def load_model(self):
        """
        This is an abstract method forcing subclasses to implement it
        """
        pass
