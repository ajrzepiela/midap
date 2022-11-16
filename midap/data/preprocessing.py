import numpy as np
import os
import skimage.io as io

from skimage import exposure
from skimage.segmentation import find_boundaries
from skimage.measure import label
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix
from typing import Optional, Union
from collections.abc import Iterable
from tqdm import tqdm

from ..utils import get_logger, set_logger_level


# ToDO:
# 1) Currently the generation of weight maps for mother machine datasets
#    is not working. The UNet for those datasets is trained without
#    a weight map for the distance to close cells.


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

    def __init__(self, n_grid=4, test_size=0.15, val_size=0.2, patch_size=128, num_split_r=10, num_split_c=10,
                 augment_patches=True, sigma=2, w_0=2, w_c0=1, w_c1=1.1, loglevel=7,
                 np_random_seed: Optional[int]=None):
        """
        Initializes the DataProcessor instance. Note that a lot parameters are used to implement the weight map
        generation according to [1] (https://arxiv.org/abs/1505.04597)
        :param n_grid: The grid used to split the original image into distinct patches for train, test and val dsets
        :param test_size: Ratio for the test set
        :param val_size: Ratio for the validation set
        :param patch_size: The size of the patches of the final images
        :param num_split_r: number of patches along row dimension for the patch generation
        :param num_split_c: number of patches along column dimension for the patch generation
        :param augment_patches: Perform data augmentation of the patches
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

        # parameters for the patch generation
        self.patch_size = patch_size
        self.num_split_r = num_split_r
        self.num_split_c = num_split_c
        self.augment_patches = augment_patches

        # sizes for test and validation set
        self.n_grid = n_grid
        self.test_size = test_size
        self.val_size = val_size

    def run(self, path_img: str, path_mask: str, path_mask_splitting_cells: Optional[str]=None):
        """
        Run the preprocessing for family machine and well plates:
            1) Load data
            2) Generate weight map
            3) Split image, masks and weight map into 4x4-grid
            4) Split patches into train and validation set
            5) Split larger patches from 4x4-grid into
                smaller ones
            6) Extract binary weight for splitting events from mask
               (1 - no splitting event, 2 - more than one)
            7) Data augmentation (if demanded)
            8) Generate binary label for coverage of colored pixels
               (1 - < 0.75, 2 - > 0.75)
            9) Return values
        :param path_img: Path to the original image used for the training
        :param path_mask: Path to the segmentation mask image (labels) corresponding to the image
        :param path_mask_splitting_cells: A mask indicating if cells are splitting, can be None
        :return: A dictionary containing the training, test and validation datasets including weight maps etc.
        """

        # 1) Load data
        self.logger.info("Loading data...")
        img = self.scale_pixel_vals(io.imread(path_img, as_gray=True))
        mask = self.scale_pixel_vals(io.imread(path_mask)).astype(int)

        # read the splitting mask
        if path_mask_splitting_cells is not None:
            mask_splitting_cells = io.imread(path_mask_splitting_cells).astype(int)
        else:
            mask_splitting_cells = None

        # 2) Generate weight map
        self.logger.info("Generating weight map...")
        weight_map = self.generate_weight_map(mask)

        # 3) Split image, masks and weight map into a grid
        self.logger.info(f"Splitting into {self.n_grid}x{self.n_grid} grid...")
        imgs, masks, weight_maps, masks_splitting_cells = self.generate_patches(
            img, mask, weight_map, mask_splitting_event=mask_splitting_cells, ensure_channel=True)

        # 4) Split patches into train and validation set
        # compute ratio of cell pixels in masks
        self.logger.info("Splitting into train and test...")
        ratio = self.compute_pixel_ratio(masks)

        # split data according to pixel coverage (ratio)
        data = self.split_data(imgs, masks, weight_maps, ratio, masks_splitting_cells)

        # 5) Split larger patches from 4x4-grid into smaller ones
        # split training patches
        self.logger.info("Generating patches...")
        # cycle through the dictionary keys and cut the patches
        for key in list(data.keys()):
            patches = []
            for img in data[key]:
                patches.append(self.cut_patches(img=img, n_row=self.num_split_r, n_col=self.num_split_c,
                                                size=self.patch_size))
            data[key] = np.concatenate(patches)

        # 6) Extract binary weight for splitting events from mask
        #    (1 - no splitting event, 2 - more than one)
        if path_mask_splitting_cells is not None:
            self.logger.info("Extract binary weight...")
            # cycle through all three instances
            keys = ["label_splitting_cells_train", "label_splitting_cells_test", "label_splitting_cells_val"]
            for key in keys:
                # get the data
                split_cells = data[key]
                split_flag = np.ones(len(split_cells))
                # flag all events
                for i, ms in enumerate(split_cells):
                    if len(np.where(ms == 127)[0]) > 0:
                        split_flag[i] = 2
                # remove old key and create new entry for the flags
                del data[key]
                new_key = key.replace("label_", "")
                data[new_key] = split_flag

        # 7) Data augmentation
        if self.augment_patches:
            self.logger.info('Data augmentation...')
            # cycle throgh train, test and val sets
            for label in ["train", "test", "val"]:
                # generate the input
                inp_data = (data[f"X_{label}"], data[f"y_{label}"], data[f"weight_maps_{label}"])
                if path_mask_splitting_cells is not None:
                    inp_data += (data[f"splitting_cells_{label}"])

                # get the output and update
                out_data = self.data_augmentation(*inp_data)
                data[f"X_{label}"], data[f"y_{label}"], data[f"weight_maps_{label}"] = out_data[:3]
                if path_mask_splitting_cells is not None:
                    data[f"splitting_cells_{label}"] = out_data[3]

        # 8) Generate binary label for coverage of colored pixels
        #    (1 - < 0.75, 2 - > 0.75)
        self.logger.info('Generating label for cell coverage')
        for label in ["train", "test", "val"]:
            # get the mean cell coverage of all patches
            ratio_cell_pixels = data[f"y_{label}"].mean(axis=(1, 2, 3))
            # normalize by the max
            ratio_cell_pixels /= ratio_cell_pixels.max()
            # get the binary flag
            data[f"ratio_cell_{label}"] = np.where(ratio_cell_pixels > 0.75, 2.0, 1.0)

        # 9) Return values
        return data

    def run_mother_machine(self, path_img: str, path_mask: str):
        """
        Run the preprocessing for mother machine:
            1) Load data
            2) Pad images and masks to adjust height and width
            3) Split images and masks into train and validation set
            4) Split each image and mask into overlapping horizontal patches
            5) Data augmentation
        :param path_img: Path to the directory containing the original images used for the training
        :param path_mask: Path to the directory containing the segmentation mask images (labels)
                          corresponding to the images
        :return: The training and validation datasets
        """

        # 1) Load data
        list_imgs = np.sort(os.listdir(path_img))
        imgs_unpad = [self.scale_pixel_vals(io.imread(path_img + i, as_gray=True)) for i in list_imgs]
        list_masks = np.sort(os.listdir(path_mask))
        masks_unpad = [self.scale_pixel_vals(io.imread(path_mask + m).astype(int)) for m in list_masks]

        # 2) Pad images and masks to adjust height and width
        height_max, width_max = self.get_max_shape(imgs_unpad)
        imgs = np.array([np.pad(i, [[0, height_max - i.shape[0]], [width_max - i.shape[1]]], mode="reflect")
                         for i in imgs_unpad])
        masks = np.array([np.pad(m, [[0, height_max - m.shape[0]], [width_max - m.shape[1]]], mode="reflect")
                          for m in masks_unpad])

        # 3) Split images and masks into train and validation set
        X_train, X_val, y_train, y_val = train_test_split(imgs, masks, test_size=self.test_size)

        # 4) Split each image and mask into horizontal patches
        height_cutoff, split_start = self.define_horizontal_splits(height=height_max)
        X_train = np.concatenate([self.horizontal_split(xtrain, height_cutoff, split_start) for xtrain in X_train])
        X_val = np.concatenate([self.horizontal_split(xval, height_cutoff, split_start) for xval in X_val])
        y_train = np.concatenate([self.horizontal_split(ytrain, height_cutoff, split_start) for ytrain in y_train])
        y_val = np.concatenate([self.horizontal_split(yval, height_cutoff, split_start) for yval in y_val])

        # 5) Data augmentation
        # currently the generation of weight maps for this dataset type
        # (mother machine) is not working. For this reason y_train_patch
        # and y_val_patch are handed to the method for the data augmentation
        # as a dummy instead of the weight maps.
        if self.augment_patches:
            X_train, y_train, _ = self.data_augmentation(X_train, y_train, y_train)
            X_val, y_val, _ = self.data_augmentation(X_val, y_val, y_val)

        return X_train, y_train, X_val, y_val

    @classmethod
    def tile_img(cls, img: np.ndarray, n_grid=4, divisor=1):
        """
        Tiles an image into a gird of n_grid x n_grid non-overlapping patches
        :param img: The image to tile must be at least two-dimensional
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

    def generate_patches(self, img: np.ndarray, mask: np.ndarray, weight_map: np.ndarray,
                         mask_splitting_event: Optional[np.ndarray]=None, ensure_channel=True):
        """
        Splits the inputs into a 4x4 grid of distinct patches
        :param img: The original input image
        :param mask: The mask for the input image
        :param weight_map: The weight map for the images
        :param mask_splitting_event: Mask for splitting events, may be None
        :param ensure_channel: Ensure that all inputs have a channel dimension
        :return: The inputs in the same order but split into patches along the first dimension
        """

        # make sure we have a channel dim
        if ensure_channel:
            img = np.atleast_3d(img)
            mask = np.atleast_3d(mask)
            weight_map = np.atleast_3d(weight_map)
            if mask_splitting_event is not None:
                mask_splitting_event = np.atleast_3d(mask_splitting_event)

        # tile everything
        img = self.tile_img(img=img, n_grid=self.n_grid)
        mask = self.tile_img(img=mask, n_grid=self.n_grid)
        weight_map = self.tile_img(img=weight_map, n_grid=self.n_grid)
        if mask_splitting_event is not None:
            mask_splitting_event = self.tile_img(img=mask_splitting_event, n_grid=self.n_grid)

        return img, mask, weight_map, mask_splitting_event

    @classmethod
    def cut_patches(cls, img: np.ndarray, n_row: int, n_col: int, size=128):
        """
        Cuts out overlapping patches from an image and returns the stack. Note that this is not a random procedure,
        it will always return the same pataches as lont as all dimensions are equal
        :param img: The image (at least two-dimensional) to cut the patches out
        :param n_row: Number of cuts along the row dimenson
        :param n_col: Number of cuts along the column dimension
        :param size: The final size of the patches (always square)
        :return: The stack of patches with dim (n_row*n_col, size, size) + img.shape[2:]
        """

        assert size < img.shape[0] and size < img.shape[1], "The size of the patches has to be smaller than the image!"

        # get the indices to cut
        row_indices = np.linspace(0, img.shape[0] - size, n_row).astype(int)
        col_indices = np.linspace(0, img.shape[0] - size, n_col).astype(int)

        # cycle and cut
        patches = []
        for i in row_indices:
            for j in col_indices:
                patches.append(img[i:i+size, j:j+size])
        return np.array(patches)

    def scale_pixel_vals(self, img: np.ndarray):
        """
        Scales pixel values between 0 and 1.
        :param img: An image that should be rescaled
        :return: The image where all pixels are between 0 and 1
        """

        img = np.array(img)
        return ((img - img.min()) / (img.max() - img.min()))

    def generate_weight_map(self, mask: np.ndarray):
        """
        Generate the weight map based on the distance to nearest and second-nearest neighbor as described in
        https://arxiv.org/abs/1505.04597
        :param mask: The mask used to generate the weights map
        :return: The generated weights map
        """
        # label cells and generate separate masks
        mask_label = label(mask)
        cell_num = np.unique(np.sort(mask_label))[1:]

        # for the weight map we need to calculate the distances from any pixel that is not part of the closest pixel
        # of any cell, where the distance is measured as Eucledean distance in pixel space
        # we start by getting the pixel ids of any pixel that is not part of any cell
        no_cell_indices = np.argwhere(mask == 0)

        # now we cycle though all cells and keep the closest distances for all cells
        dists = []
        self.logger.info("Calculating pixel distances...")
        for cell_id in tqdm(cell_num):
            # isolate the cell
            cell_mask = (mask_label == cell_id).astype(int)
            # get the boundaries (boolean array)
            bounds = find_boundaries(cell_mask, mode='inner')
            # get the indices of the boundary
            indices = np.argwhere(bounds)
            # calculate all the distances, keep only the smallest
            dist = np.min(distance_matrix(no_cell_indices, indices), axis=1)
            # append
            dists.append(dist)

        # get distance to nearest and second-nearest cell
        self.logger.info("Calculating weights...")
        min_dist_sort = np.sort(dists, axis=0)
        d1_val = min_dist_sort[0, :]
        d2_val = min_dist_sort[1, :]

        # calculate the weights
        weights = self.w_c0 + self.w_0 * \
            np.exp((-1 * (d1_val + d2_val) ** 2) / (2 * (self.sigma ** 2)))
        # create the map with channel dim, use default weight c1
        w = np.full(mask.shape + (1, ), fill_value=self.w_c1)
        # assign the rest
        w[no_cell_indices[:,0], no_cell_indices[:,1], 0] = weights

        return w

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

    def split_data(self, imgs: np.ndarray, masks: np.ndarray, weight_maps: np.ndarray, ratio: np.ndarray,
                   label_splitting_cells: Optional[np.ndarray]=None):
        """
        Split data depending on the ratio of colored pixels in all images (patches).
        :param imgs: The original images split into patches (training data)
        :param masks: The masks corresponding to the images (labels)
        :param weight_maps: The weight maps used for the loss (weights)
        :param ratio: A 1D array containing the ratios pixels containing cells used for the stratification (balancing)
        :param label_splitting_cells: Optional array containing the labels of splitting cells
        :return: A dictionary containing the train, test, val splits of the input arrays
        """

        # get the number of images in the test set, this defines the number of "classes"
        n_test = int(self.test_size*len(ratio)) + 1
        # get the stratification according to quantiles
        stratification = self.get_quantile_classes(ratio, n_test)

        # input to split into sets
        arrays = (ratio, imgs, masks, weight_maps)
        if label_splitting_cells is not None:
            arrays += (label_splitting_cells, )

        # split
        ratio_train, ratio_test, *splits = train_test_split(*arrays, test_size=self.test_size, stratify=stratification)

        # add to result dictionarray
        res = {"X_train": splits[0], "X_test": splits[1],
               "y_train": splits[2], "y_test": splits[3],
               "weight_maps_train": splits[4], "weight_maps_test": splits[5]}
        if label_splitting_cells is not None:
            res["label_splitting_cells_train"] = splits[6]
            res["label_splitting_cells_test"] = splits[7]

        # we split the training set into training and validation
        n_val = int(self.val_size*len(ratio_train)) + 1
        # get the stratification according to quantiles
        stratification = self.get_quantile_classes(ratio_train, n_val)

        # input to split into sets
        arrays = (res["X_train"], res["y_train"], res["weight_maps_train"])
        if label_splitting_cells is not None:
            arrays += (res["label_splitting_cells_train"], )

        # split
        splits = train_test_split(*arrays, test_size=self.val_size, stratify=stratification)

        # add to result dictionarray
        res.update({"X_train": splits[0], "X_val": splits[1],
                    "y_train": splits[2], "y_val": splits[3],
                    "weight_maps_train": splits[4], "weight_maps_val": splits[5]})
        if label_splitting_cells is not None:
            res["label_splitting_cells_train"] = splits[6]
            res["label_splitting_cells_val"] = splits[7]

        return res

    def define_horizontal_splits(self, height: int, num_splits=150):
        """
        Define the starting heights of the horizontal splits. The height of the resulting horizontal patch is quarter
        the height of the original image. The width of the horizontal patch is the width of the original image.
        :param height: The height of the images that should be split
        :param num_splits: The number of splits to perform
        :return: The height cutoff (quarter of the input height) and the starting points of the splits
        """

        # define the minimum height (split_start) of random splits
        # random patch should have the quarter height of the original image
        height_cutoff = height // 4
        split_start = np.linspace(0, height, num_splits, dtype=int)

        return height_cutoff, split_start

    def horizontal_split(self, img: np.ndarray, height_cutoff: int, split_start: Iterable[float]):
        '''
        Split an image into random horizontal patches accoring to
        previously defined starting heights of the horizontal splits. The image is
        horizontally mirrored before the split to to use the full height of the image.
        '''

        # check if the image is wide enough
        if img.shape[0] < np.max(split_start) + height_cutoff:
            img = np.pad(img, pad_width=[0, np.max(split_start) + height_cutoff - img.shape[0],
                                        [0, 0]], mode="reflect")

        # split image according to where split starts
        hor_splits = []
        for start in split_start:
            hor_splits.append(img[start:(start + height_cutoff), :])

        return np.array(hor_splits)

    def get_max_shape(self, imgs: Iterable[np.ndarray], divisor=16):
        """
        Get the maximal height and width across all images in a dataset. The maximal height and width are also set to
        the next higher multiple of divisor to allow an even split of feature maps in the unet.
        :param imgs: A list or iterable of 2D arrays
        :param divisor: The divisor for the largest shape
        :return: The largest possible shape that is divisible by divisor and smaller or equal than the
                 largest shape in imgs
        """

        imgs_shape = [i.shape for i in imgs]
        height_max = np.ceil(np.max([s[0] for s in imgs_shape]) / divisor).astype(int) * divisor
        width_max = np.ceil(np.max([s[1] for s in imgs_shape]) / divisor).astype(int) * divisor

        return height_max, width_max

    def data_augmentation(self, imgs: np.ndarray, masks: np.ndarray, weight_maps: np.ndarray,
                          splitting_cells: Optional[np.ndarray]=None, shuffle=True):
        """
        Augments list of imgs, masks and weights maps by vertical and horizontal flips and increase and decrease
        of brightness.
        :param imgs: Array of images
        :param masks: Array of masks (labels) corresponding to the images
        :param weight_maps: Array of weight maps (for loss) corresponding to the images
        :param splitting_cells: 1D array with the same length as the images indicating if a split event happens
        :param shuffle: If True (default) shuffle the output before returning
        :return: The original inputs with all augmentations
        """

        # init emtpy list to collect the augmented images
        aug_imgs = []
        aug_masks = []
        aug_weight_maps = []
        if splitting_cells is not None:
            aug_splitting_cells = []

        for i, (img, mask, weight_map) in enumerate(zip(imgs, masks, weight_maps)):
            # the original image
            aug_imgs.append(img)
            aug_masks.append(mask)
            aug_weight_maps.append(weight_map)
            if splitting_cells is not None:
                aug_splitting_cells.append(splitting_cells[i])

            # vertical flip
            aug_imgs.append(np.flipud(img))
            aug_masks.append(np.flipud(mask))
            aug_weight_maps.append(np.flipud(weight_map))
            if splitting_cells is not None:
                aug_splitting_cells.append(splitting_cells[i])

            # horizontal flip
            aug_imgs.append(np.fliplr(img))
            aug_masks.append(np.fliplr(mask))
            aug_weight_maps.append(np.fliplr(weight_map))
            if splitting_cells is not None:
                aug_splitting_cells.append(splitting_cells[i])

            # decrease brightness
            for gamma in [0.4, 0.6, 0.8, 1.2, 1.4, 1.6]:
                aug_imgs.append(exposure.adjust_gamma(img, gamma=gamma, gain=1))
                aug_masks.append(mask)
                aug_weight_maps.append(weight_map)
                if splitting_cells is not None:
                    aug_splitting_cells.append(splitting_cells[i])

        # create a shuffle index if necessary
        if shuffle:
            perm = np.random.permutation(len(aug_imgs))
        else:
            perm = Ellipsis

        # create the output
        out = (np.array(aug_imgs)[perm], np.array(aug_masks)[perm], np.array(aug_weight_maps)[perm])
        if splitting_cells is not None:
            out += (np.array(aug_splitting_cells)[perm])

        return out
