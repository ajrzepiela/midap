import numpy as np
import math
import os
import re

import matplotlib
# backend has to be set before the pyplot import, TkAgg is compatible with most clusters
matplotlib.use('TkAgg')

import skimage.io as io
from skimage.registration import phase_cross_correlation

from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Iterable

from ..utils import get_logger

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = int(os.environ["__VERBOSE"])
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)

class CutoutImage(ABC):
    """
    A class that performs the image cutout for the different channels
    """

    # this logger will be shared by all instances and subclasses
    logger = logger

    def __init__(self, paths: Iterable[str], config):


        # if paths is just a single string we pack it into a list
        if isinstance(paths, str):
            self.paths = [paths]
        else:
            self.paths = paths

        # Set the config file
        self.config = config

        # get the file lists
        self.channels = [[os.path.join(channel, f) for f in sorted(os.listdir(channel))] for channel in self.paths]

        # adjust the length such that all channels have the same number of elements
        self.min_frames = np.min([len(channel) for channel in self.channels])
        for i in range(len(self.channels)):
            self.channels[i] = self.channels[i][:self.min_frames]

    def align_two_images(self, src_img: np.ndarray, ref_img: np.ndarray):
        """
        Calculates the shifts necessary to align to images
        :param src_img: Source image
        :param ref_img: Reference image
        :returns: shifts as vector for the alignment
        """
        # aligns a source image in comparison to a reference image
        return phase_cross_correlation(src_img, ref_img, normalization=None)[0].astype(int)

    def align_all_images(self):
        """
        Calculates the shifts necessary to align all images
        """
        # load 1st image of phase channel
        files = self.channels[0]
        src = self.open_tiff(files[0])
        self.shifts = []
        for i in tqdm(range(1,len(files))):
            ref = self.open_tiff(files[i])
            # align image compared to 1st image
            shift = self.align_two_images(src, ref)
            self.shifts.append(shift)

        # computes the offset to choose depending on shift between images
        self.off = int(math.ceil(np.max(np.abs(self.shifts)) / 10.0)) * 10

    def save_corners(self):
        """
        Saves the corners for the cutout into the settings.sh
        """
        # write the corners into the settings.sh file
        with open("settings.sh", mode="r+") as f:
            # read the file content
            content = f.read()

            if "CORNERS=" in content:
                # we replace the variable
                content = re.sub(f"CORNERS\=.*", f"CORNERS={self.corners_cut}".replace(",", ""), content)

                # truncate, set stream to start and write
                f.truncate(0)
                f.seek(0)
                f.write(content)
            else:
                # we add a new line to the file
                # replace commas to create a bash array
                f.write(f"CORNERS={self.corners_cut}\n".replace(",", ""))

    def open_tiff(self, path):
        """
        Opens a tiff file and returns the images as array
        :param path: path to the file
        :returns: array of images
        """
        # load all pages of tiff file and return list of image arrays
        im = io.imread(path)
        return im

    def do_cutout(self, img, corners_cut):
        """
        Performs a cutout of an image
        :param img: Image ad array
        :param corners_cut: The corners used for the cutout
        :returns: The cutout from the image given the corners
        """
        # generate cutout of image
        left_x, right_x, lower_y, upper_y = corners_cut
        cutout = img[lower_y:upper_y, left_x:right_x]
        return cutout

    def scale_pixel_val(self, img):
        """
        Rescale the pixel values of the image
        :param img: The input image as array
        :returns: The images with pixels scales to standard RGB values
        """
        img_scaled = (255 * ((img - np.min(img))/np.max(img - np.min(img)))).astype('uint8')
        return img_scaled

    def shift_image(self, im, shift):
        """
        Aligns an image of an additional channels by shift
        :param im: image to shift as array
        :param shift: The shift to perform
        :returns: The shifted image
        """
        if self.off-shift[0] != 0:
            im = im[(self.off - shift[0]):(-self.off - shift[0]), :]
        if self.off - shift[1] != 0:
            im = im[:, (self.off - shift[1]):(-self.off - shift[1])]
        return im

    def save_cutout(self, files, file_names):
        """
        Saves the cutouts into the proper directory
        :param files: A list of arrrays (the cutouts) to save
        :param file_names: The list of file names from the original files
        """
        # save of cutouts
        dir_name = os.path.dirname(os.path.dirname(file_names[0]))
        for f, i in zip(file_names, files):
            fname = f"{os.path.splitext(os.path.basename(f))[0]}_cut.png"
            io.imsave(os.path.join(dir_name, 'cut_im', fname), i, check_contrast=False)
        
    def run_align_cutout(self):
        """
        Aligns and cut out all images from all channels
        """

        self.logger.info('Aligning images...')
        self.align_all_images()
        
        self.logger.info('Cutting images...')
        # cycle through the channels
        for channel_id, files in enumerate(self.channels):
            self.logger.info(f'Starting with channel {channel_id+1}/{len(self.channels)}')
            # list for the aligned cutouts
            aligned_cutouts = []

            # get the first image
            src = self.open_tiff(files[0])
            # offset of 1st image
            if self.off == 0:
                src_off = src
            else:
                src_off = src[self.off:-self.off, self.off:-self.off]

            # We get the corners using the PH channel
            if channel_id == 0:
                # set the corner to cut
                self.cut_corners(img=src_off)

                # save the corner to file
                self.save_corners()

            # perform the cutout of the first image
            cutout = self.do_cutout(src_off, self.corners_cut)
            # scale the pixel values
            cut_src = self.scale_pixel_val(cutout)

            # add to list
            aligned_cutouts.append(cut_src)

            # cutout of all other images of all channels
            for i in tqdm(range(1, len(files))):
                ref = self.open_tiff(files[i])

                # align and cutout phase image compared to 1st image
                aligned_img = self.shift_image(ref, self.shifts[i-1])
                cut_img = self.do_cutout(aligned_img, self.corners_cut)
                # sacle the pixel values
                proc_img = self.scale_pixel_val(cut_img)
                aligned_cutouts.append(proc_img)

            self.save_cutout(aligned_cutouts, files)

    @abstractmethod
    def cut_corners(self, img):
        """
        This is an abstract method forcing subclasses to implement it
        """
        pass
