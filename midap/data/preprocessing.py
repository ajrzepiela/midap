import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import skimage.io as io
from numba import njit
from skimage.measure import label
from skimage.segmentation import find_boundaries
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..utils import get_logger, set_logger_level


class DataProcessor(object):
    """
    Preprocessing of the raw images, the masks and masks for splitting events.

    a) The preprocessing for family machine and well plates:
        1) patch generation by forming a 4x4-grid
        2) split of patches into train and validation set
        3) random patch generation from train and
            validation set
        4) data augmentation (horizontal and vertical flip
            plus increase and decrease of brightness)
    b) The preprocessing for the mother machine:
        1) split of training images and masks into train and validation set
        2) random patch generation from train and
            validation set
        3) data augmentation (horizontal and vertical flip
            plus increase and decrease of brightness)
    """

    # This logger can be accessed by classmethods etc
    logger = get_logger(__file__)

    def __init__(self, paths: Union[str, bytes, os.PathLike, List[Union[str, bytes, os.PathLike]]],
                 n_grid=4, test_size=0.15, val_size=0.2, sigma=2.0, w_0=2.0, w_c0=1.0, w_c1=1.1, loglevel=7,
                 np_random_seed: Optional[int]=None):
        """
        Initializes the DataProcessor instance. Note that a lot parameters are used to implement the weight map
        generation according to [1] (https://arxiv.org/abs/1505.04597)
        :param paths: A single path or a list of paths to files that can be used for training. Only the raw images
                      have to be listed. However, it is expected that the raw images end with "*raw.tif" and a
                      corresponding segmentations mask "*seg.tif" exists.
        :param n_grid: The grid used to split the original image into distinct patches for train, test and val dsets
        :param test_size: Ratio for the test set
        :param val_size: Ratio for the validation set
        :param sigma: sigma parameter used for the weight map calculation [1]
        :param w_0: w_0 parameter used for the weight map calculation [1]
        :param w_c0: basic class weight for non-cell pixel parameter used for the weight map calculation [1]
        :param w_c1: basic class weight for cell pixel parameter used for the weight map calculation [1]
        :param loglevel: The loglevel of the logger instance, 0 -> no output, 7 (default) -> max output
        :param np_random_seed: A random seed for the numpy random seed generator, defaults to None, which will lead
                               to non-reproducible behaviour. Note that the state will be set at initialisation and
                               not reset by any of the methods.
        """

        # set the log level
        set_logger_level(self.logger, loglevel)

        # set the random seed
        if np_random_seed is not None:
            self.logger.info(f"Setting numpy random seed to: {np_random_seed}")
            np.random.seed(np_random_seed)

        # parameters for the computation of the weight maps
        self.sigma = sigma
        self.w_0 = w_0
        self.w_c0 = w_c0
        self.w_c1 = w_c1

        # sizes for test and validation set
        self.n_grid = n_grid
        self.test_size = test_size
        self.val_size = val_size

        # training data
        if isinstance(paths, list):
            self.img_paths = [Path(p) for p in paths]
        else:
            self.img_paths = [Path(paths)]
        self.seg_paths = [p.parent.joinpath(p.name.replace("raw.tif", "seg.tif")) for p in self.img_paths]

        # check for existence
        for raw, seg in zip(self.img_paths, self.seg_paths):
            if not raw.name.endswith("raw.tif"):
                raise ValueError(f"Raw image name does not match format: {raw} (needs to end with raw.tif)")
            if not raw.exists():
                raise FileNotFoundError(f"File for training does not exist: {raw}")
            if not seg.exists():
                raise FileNotFoundError(f"File for training does not exist: {seg}")

    def get_dset(self):
        """
        Run the preprocessing for family machine and well plates:
            1) Load data
            2) Generate weight map
            3) Split images, masks and weight map into a grid
            4) Split into training and validation dataset
        :return: A dictionary containing the training, test and validation datasets including weight maps etc.
        """

        # a default dict for the full data
        full_data = defaultdict(list)
        for img_path, seg_path in zip(self.img_paths, self.seg_paths):
            # 1) Load data
            self.logger.info(f"Loading file {img_path}")
            img = self.scale_pixel_vals(io.imread(img_path, as_gray=True))
            seg = self.scale_pixel_vals_seg(io.imread(seg_path)).astype(int)

            # 2) Generate weight map
            w_string = f"w_0={self.w_0}_w_c0={self.w_c0}_w_c1={self.w_c1}_sigma={self.sigma}"
            w_path = img_path.parent.joinpath(img_path.name.replace("raw.tif", f"{w_string}_weights.tif"))
            if w_path.exists():
                weights = io.imread(w_path)
            else:
                self.logger.info("Generating weight map...")
                weights = self.generate_weight_map(seg)
                io.imsave(w_path, weights)

            # 3) Split image, masks and weight map into a grid
            self.logger.info(f"Splitting into {self.n_grid}x{self.n_grid} grid...")
            imgs, masks, weight_maps = self.generate_patches(img, seg, weights, ensure_channel=True)

            # 4) Split patches into train and validation set
            # compute ratio of cell pixels in masks
            self.logger.info("Splitting into train and test...")
            ratio = self.compute_pixel_ratio(masks)

            # split data according to pixel coverage (ratio)
            data = self.split_data(imgs, masks, weight_maps, ratio)

            # update the full data
            for key, val in data.items():
                full_data[key].append(val)

        # 5) Return values
        return full_data

    @classmethod
    def tile_img(cls, img: np.ndarray, n_grid=4, divisor=1):
        """
        Tiles an image into a gird of n_grid x n_grid non-overlapping patches
        :param img: The image, must be at least two-dimensional
        :param n_grid: The grid dimension for the tiling
        :param divisor: Ensure that all the dimensions of the tiles are divisible by divisor, e.g. divisor=2 to have
                        even dimensions. Note that this may cause a large portion of the original image to be
                        thrown away
        :return: The tiles of dimension (n_grid x n_grid, img.shape[0]//4, img.shape[1]//4) + img.shape[2:]
        """
        # get the shape img
        height, width, *_ = img.shape

        # get the dims of the tiles (make sure they are divisible by divisor)
        r_dim = divisor*(height//(n_grid*divisor))
        c_dim = divisor*(width//(n_grid*divisor))

        assert r_dim != 0 and c_dim != 0, f"The requested tiling causes at least on dimension of the tiles to be 0: " \
                                          f"{r_dim=} {c_dim=}"

        # tile the image
        tiles = []
        for i in range(n_grid):
            for j in range(n_grid):
                tiles.append(img[i*r_dim:(i+1)*r_dim, j*c_dim:(j+1)*c_dim])

        # transform to array and return
        return np.array(tiles)

    def generate_patches(self, img: np.ndarray, mask: np.ndarray, weight_map: np.ndarray, ensure_channel=True):
        """
        Splits the inputs into a grid of distinct patches
        :param img: The original input image
        :param mask: The mask for the input image
        :param weight_map: The weight map for the images
        :param ensure_channel: Ensure that all inputs have a channel dimension
        :return: The inputs in the same order but split into patches along the first dimension
        """

        # make sure we have a channel dim
        if ensure_channel:
            img = np.atleast_3d(img)
            mask = np.atleast_3d(mask)
            weight_map = np.atleast_3d(weight_map)

        # tile everything
        img = self.tile_img(img=img, n_grid=self.n_grid)
        mask = self.tile_img(img=mask, n_grid=self.n_grid)
        weight_map = self.tile_img(img=weight_map, n_grid=self.n_grid)

        return img, mask, weight_map

    @classmethod
    def scale_pixel_vals(cls, img: np.ndarray):
        """
        Scales pixel values between 0 and 1.
        :param img: An image that should be rescaled
        :return: The image where all pixels are between 0 and 1
        """

        img = np.array(img)
        return ((img - img.min()) / (img.max() - img.min()))
    
    @classmethod
    def scale_pixel_vals_seg(cls, seg: np.ndarray):
        """
        Converts all segmentation images (labeled and binary) to binary image.
        :param seg: A segmentation image that should be converted
        :return: The binary segmentation
        """

        seg = (seg > 0).astype(int)
        return seg

    @staticmethod
    @njit("void(i8[:,:],f8[:,:,:],i4)")
    def update_dist_array(bound_indices: np.ndarray, dist_array: np.ndarray, cut_off: int):
        """
        This is a numba optimized static method to update the distance array to calculate the weights of a map
        It works by defining a window around the current cell (with width cut_off) and then calculating the minimum
        distance of each pixel inside this window to the cell bounds. Afterwards the distance array is updated with
        this new minimal distance if necessary
        :param bound_indices: The indices of the cell bounds 2D (N, 2) array with int dtype
        :param dist_array: The full array of distances with shape (n, m, 2) where (n, m) is the shape of the original
                           image containing all the cells. [i,j,0] contains the smallest distance from that pixel
                           to the next cell boundary, and [i,j,1] the next smallest distance to another cell
        :param cut_off: The size of the cutoff window
        :return:
        """

        # get the shape of the original image
        n, m, _ = dist_array.shape

        # define the window around the current cell
        min_x = max(min(bound_indices[:,0]) - cut_off, 0)
        max_x = min(max(bound_indices[:,0]) + cut_off, n)
        min_y = max(min(bound_indices[:,1]) - cut_off, 0)
        max_y = min(max(bound_indices[:,1]) + cut_off, m)

        # cycle through all pixels in the window
        for id_x in range(min_x, max_x):
            for id_y in range(min_y, max_y):
                # the current smallest distance
                tmp_min = np.inf
                # cycle through all bounds
                for bound in bound_indices:
                    bound_x = bound[0]
                    bound_y = bound[1]
                    # calculate distance and update temp min if necessary
                    d = np.sqrt((id_x - bound_x)**2 + (id_y - bound_y)**2)
                    if d < tmp_min:
                        tmp_min = d
                # update the dist array if necessary
                if tmp_min < dist_array[id_x, id_y, 0]:
                    # second smallest update
                    dist_array[id_x, id_y, 1] = dist_array[id_x, id_y, 0]
                    # smallest update
                    dist_array[id_x, id_y, 0] = tmp_min
                elif tmp_min < dist_array[id_x, id_y, 1]:
                    # second smallest update
                    dist_array[id_x, id_y, 1] = tmp_min

    def generate_weight_map(self, mask: np.ndarray, cut_off=10):
        """
        Generate the weight map based on the distance to nearest and second-nearest neighbor as described in
        https://arxiv.org/abs/1505.04597
        :param mask: The mask used to generate the weights map
        :param cut_off: Only consider a region of cut_off*self.sigma around each border pixel weight, the weight. Since
                        the weight is porportional to exp(-d**2/sigma**2) -> 10 sigma will be a weight of ~0
        :return: The generated weights map
        """
        # label cells and generate separate masks
        mask_label = label(mask)
        cell_num = np.unique(np.sort(mask_label))[1:]

        # for the weight map we need to calculate the distances from any pixel that is not part of the closest pixel
        # of any cell, where the distance is measured as Eucledean distance in pixel space
        # we start by initializeing the distance array (default large -> weight ~0)
        dist_array = np.full(shape=mask.shape + (2, ), fill_value=10000.0)

        # now we cycle though all cells and keep the closest distances for all cells
        self.logger.info("Calculating pixel distances...")
        for cell_id in tqdm(cell_num):
            # isolate the cell
            cell_mask = (mask_label == cell_id).astype(int)
            # get the boundaries (boolean array)
            bounds = find_boundaries(cell_mask, mode='inner')
            # get the indices of the boundary
            indices = np.argwhere(bounds)
            # update the dist array
            self.update_dist_array(bound_indices=indices, dist_array=dist_array, cut_off=int(cut_off*self.sigma))

        # get distance to nearest and second-nearest cell
        self.logger.info("Calculating weights...")
        d1_val = dist_array[...,0]
        d2_val = dist_array[...,1]

        # calculate the weights
        weights = self.w_c0 + self.w_0 * \
            np.exp((-1 * (d1_val + d2_val) ** 2) / (2 * (self.sigma ** 2)))
        # where there is a cell, we want to have c1
        weights[mask != 0] = self.w_c1

        return weights

    @classmethod
    def compute_pixel_ratio(self, masks: np.ndarray):
        """
        Compute the ratio of colored pixels in a mask.
        :param masks: An array of masks, at least two-dimensional
        :return: The pixel ratios accumulated over all dimensions besides the first
        """

        # get the counts
        counts = np.array([np.count_nonzero(m) for m in masks])
        # get the size of the counts
        total = np.prod(masks.shape[1:])

        # return the ratio
        return counts / total

    @classmethod
    def get_quantile_classes(cls, x: np.ndarray, n: int):
        """
        Creates an array with the same length as x where each element is labeled. The number of labels is given by
        n. x is sorted and split into n different part according to np.array_split, each part gets its own label.
        :param x: 1D array of entries to label
        :param n: Total number of labels
        :return: The classes of the elements from x
        """

        # sort the ratios
        asort = np.argsort(x)
        # we bundle x in "quantiles" for the stratification
        stratification = np.zeros(x.shape, dtype=int)
        for class_id, indices in enumerate(np.array_split(asort, indices_or_sections=n)):
            stratification[indices] = class_id

        return stratification

    def split_data(self, imgs: np.ndarray, masks: np.ndarray, weight_maps: np.ndarray, ratio: np.ndarray):
        """
        Split data depending on the ratio of colored pixels in all images (patches).
        :param imgs: The original images split into patches (training data)
        :param masks: The masks corresponding to the images (labels)
        :param weight_maps: The weight maps used for the loss (weights)
        :param ratio: A 1D array containing the ratios pixels containing cells used for the stratification (balancing)
        :return: A dictionary containing the train, test, val splits of the input arrays
        """

        # get the number of images in the test set, this defines the number of "classes"
        n_test = int(self.test_size*len(ratio)) + 1
        # get the stratification according to quantiles
        stratification = self.get_quantile_classes(ratio, n_test)

        # input to split into sets
        arrays = (ratio, imgs, masks, weight_maps)

        # split
        ratio_train, ratio_test, *splits = train_test_split(*arrays, test_size=self.test_size, stratify=stratification)

        # add to result dictionarray
        res = {"X_train": splits[0], "X_test": splits[1],
               "y_train": splits[2], "y_test": splits[3],
               "weight_maps_train": splits[4], "weight_maps_test": splits[5]}

        # we split the training set into training and validation
        n_val = int(self.val_size*len(ratio_train)) + 1
        # get the stratification according to quantiles
        stratification = self.get_quantile_classes(ratio_train, n_val)

        # input to split into sets
        arrays = (res["X_train"], res["y_train"], res["weight_maps_train"])

        # split
        splits = train_test_split(*arrays, test_size=self.val_size, stratify=stratification)

        # add to result dictionarray
        res.update({"X_train": splits[0], "X_val": splits[1],
                    "y_train": splits[2], "y_val": splits[3],
                    "weight_maps_train": splits[4], "weight_maps_val": splits[5]})

        return res
