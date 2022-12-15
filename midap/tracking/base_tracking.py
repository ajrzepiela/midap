import skimage.io as io
from skimage.transform import resize

import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from ..utils import get_logger

import os
import psutil
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

    def __init__(self, imgs: Iterable[str], 
                        segs: Iterable[str]=None, 
                        model_weights: Optional[str]=None, 
                        input_size: Optional[tuple]=None,
                        target_size: Optional[tuple]=None, 
                        crop_size: Optional[str]=None):
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


    def load_data(self, cur_frame: int):
        """
        Loads and resizes raw images and segmentation images of the previous and current time frame.
        :param cur_frame: Number of the current frame.
        :return: The loaded and resized images of the current frame, the previous frame, the current segmentation and
                the prvious segmentation
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

 
    def run_tracking(self, *args, **kwargs): #logger, output_folder
        """
        Track all frames using cropped images as input.
        """

        self.track_all_frames(*args, **kwargs)

        #self.store_data()


    @abstractmethod
    def track_all_frames(self, *args, **kwargs):
        """
        This is an abstract method forcing subclasses to implement it
        """
        pass
