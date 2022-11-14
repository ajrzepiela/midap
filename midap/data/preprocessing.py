import numpy as np
import os
import operator
import skimage.io as io

from functools import reduce
from skimage import exposure
from skimage.segmentation import find_boundaries
from skimage.measure import label
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from scipy.spatial import distance_matrix
from tqdm import tqdm

from ..utils import get_logger


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

    def __init__(self, sigma=2, w_0=2, w_c0=1, w_c1=1.1, num_split_r=10,
                 num_split_c=10, test_size=0.15, val_size=0.2, loglevel=7):

        # get the logger
        self.logger = get_logger(__file__, loglevel)

        # parameters for the computation of the weight maps
        self.sigma = sigma  # params for dist matrix
        self.w_0 = w_0  # params for dist matrix
        self.w_c0 = w_c0  # weight of background pixels
        self.w_c1 = w_c1  # weight of foreground pixels

        # parameters for the patch generation
        self.num_split_r = num_split_r  # number of patches along x/y-axis
        self.num_split_c = num_split_c  # number of patches along x/y-axis

        # sizes for
        self.test_size = test_size
        self.val_size = val_size

    def run(self, path_img, path_mask, path_mask_splitting_cells=None):
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
            7) Data augmentation
            8) Generate binary label for coverage of colored pixels
               (1 - < 0.75, 2 - > 0.75)
            9) Return values
        """

        # 1) Load data
        self.logger.info("Loading data...")
        img = self.scale_pixel_vals(io.imread(path_img, as_gray=True))
        mask = self.scale_pixel_vals(io.imread(path_mask)).astype(int)

        # read the splitting mask
        if path_mask_splitting_cells:
            mask_splitting_cells = \
                io.imread(path_mask_splitting_cells).astype(int)

        # 2) Generate weight map
        self.logger.info("Generating weight map...")
        weight_map = self.generate_weight_map(mask)

        # 3) Split image, masks and weight map into 4x4-grid
        self.logger.info('Split into 4x4 grid')
        if path_mask_splitting_cells:
            imgs, masks, weight_maps, masks_splitting_cells = self.generate_patches(
                img, mask, weight_map, mask_splitting_cells)
        else:
            imgs, masks, weight_maps = self.generate_patches(
                img, mask, weight_map)

        # 4) Split patches into train and validation set
        # compute ratio of cell pixels in masks
        print('Split into train and test')
        ratio = self.compute_pixel_ratio(masks)

        # split data according to pixel coverage (ratio)
        if path_mask_splitting_cells:
            (X_train,
             y_train,
             weight_maps_train,
             masks_splitting_cells_train,
             X_val,
             y_val,
             weight_maps_val,
             masks_splitting_cells_val,
             X_test,
             y_test,
             weight_maps_test,
             masks_splitting_cells_test) = self.split_data(imgs,
                                                           masks,
                                                           weight_maps,
                                                           ratio,
                                                           masks_splitting_cells)
        else:
            (X_train, y_train, weight_maps_train, X_val, y_val,
             weight_maps_val, X_test, y_test,
             weight_maps_test) = self.split_data(imgs, masks, weight_maps, ratio)

        # 5) Split larger patches from 4x4-grid into smaller ones
        # split training patches
        print('Generate random patches')

        # split train patches
        X_train_rand_patches = []
        y_train_rand_patches = []
        weight_maps_train_rand_patches = []
        if path_mask_splitting_cells:
            masks_splitting_cells_train_rand_patches = []
            for i in range(len(X_train)):
                (
                    X_train_rand,
                    y_train_rand,
                    weight_maps_train_rand,
                    masks_splitting_cells_train_rand) = self.generate_random_patches(
                    X_train[i],
                    y_train[i],
                    weight_maps_train[i],
                    masks_splitting_cells_train[i])
                X_train_rand_patches.append(X_train_rand)
                y_train_rand_patches.append(y_train_rand)
                weight_maps_train_rand_patches.append(weight_maps_train_rand)
                masks_splitting_cells_train_rand_patches.append(
                    masks_splitting_cells_train_rand)
            masks_splitting_cells_train_rand_patches = np.concatenate(
                masks_splitting_cells_train_rand_patches)
        else:
            for i in range(len(X_train)):
                (X_train_rand, y_train_rand, weight_maps_train_rand) = self.generate_random_patches(
                    X_train[i], y_train[i], weight_maps_train[i])
                X_train_rand_patches.append(X_train_rand)
                y_train_rand_patches.append(y_train_rand)
                weight_maps_train_rand_patches.append(weight_maps_train_rand)
        X_train_rand_patches = np.concatenate(X_train_rand_patches)
        y_train_rand_patches = np.concatenate(y_train_rand_patches)
        weight_maps_train_rand_patches = np.concatenate(
            weight_maps_train_rand_patches)

        # split val patches
        X_val_rand_patches = []
        y_val_rand_patches = []
        weight_maps_val_rand_patches = []
        if path_mask_splitting_cells:
            masks_splitting_cells_val_rand_patches = []
            for i in range(len(X_val)):
                (X_val_rand,
                 y_val_rand,
                 weight_maps_val_rand,
                 masks_splitting_cells_val_rand) = self.generate_random_patches(X_val[i],
                                                                                y_val[i],
                                                                                weight_maps_val[i],
                                                                                masks_splitting_cells_val[i])
                X_val_rand_patches.append(X_val_rand)
                y_val_rand_patches.append(y_val_rand)
                weight_maps_val_rand_patches.append(weight_maps_val_rand)
                masks_splitting_cells_val_rand_patches.append(
                    masks_splitting_cells_val_rand)
            masks_splitting_cells_val_rand_patches = np.concatenate(
                masks_splitting_cells_val_rand_patches)
        else:
            for i in range(len(X_val)):
                (X_val_rand, y_val_rand, weight_maps_val_rand) = self.generate_random_patches(
                    X_val[i], y_val[i], weight_maps_val[i])
                X_val_rand_patches.append(X_val_rand)
                y_val_rand_patches.append(y_val_rand)
                weight_maps_val_rand_patches.append(weight_maps_val_rand)
        X_val_rand_patches = np.concatenate(X_val_rand_patches)
        y_val_rand_patches = np.concatenate(y_val_rand_patches)
        weight_maps_val_rand_patches = np.concatenate(
            weight_maps_val_rand_patches)

        # split test patches
        X_test_rand_patches = []
        y_test_rand_patches = []
        weight_maps_test_rand_patches = []
        if path_mask_splitting_cells:
            masks_splitting_cells_test_rand_patches = []
            for i in range(len(X_test)):
                (X_test_rand,
                 y_test_rand,
                 weight_maps_test_rand,
                 masks_splitting_cells_test_rand) = self.generate_random_patches(X_test[i],
                                                                                 y_test[i],
                                                                                 weight_maps_test[i],
                                                                                 masks_splitting_cells_test[i])
                X_test_rand_patches.append(X_test_rand)
                y_test_rand_patches.append(y_test_rand)
                weight_maps_test_rand_patches.append(weight_maps_test_rand)
                masks_splitting_cells_test_rand_patches.append(
                    masks_splitting_cells_test_rand)
            masks_splitting_cells_test_rand_patches = np.concatenate(
                masks_splitting_cells_test_rand_patches)
        else:
            for i in range(len(X_test)):
                (X_test_rand, y_test_rand, weight_maps_test_rand) = self.generate_random_patches(
                    X_test[i], y_test[i], weight_maps_test[i])
                X_test_rand_patches.append(X_test_rand)
                y_test_rand_patches.append(y_test_rand)
                weight_maps_test_rand_patches.append(weight_maps_test_rand)
        X_test_rand_patches = np.concatenate(X_test_rand_patches)
        y_test_rand_patches = np.concatenate(y_test_rand_patches)
        weight_maps_test_rand_patches = np.concatenate(
            weight_maps_test_rand_patches)

        # 6) Extract binary weight for splitting events from mask
        #    (1 - no splitting event, 2 - more than one)
        print('Extract binary weight')
        if path_mask_splitting_cells:
            splitting_cells_train = np.ones(
                len(masks_splitting_cells_train_rand_patches))
            for i, ms in enumerate(masks_splitting_cells_train_rand_patches):
                if len(np.where(ms == 127)[0]) > 0:
                    splitting_cells_train[i] = 2

            splitting_cells_val = np.ones(
                len(masks_splitting_cells_val_rand_patches))
            for i, ms in enumerate(masks_splitting_cells_val_rand_patches):
                if len(np.where(ms == 127)[0]) > 0:
                    splitting_cells_val[i] = 2

            splitting_cells_test = np.ones(
                len(masks_splitting_cells_test_rand_patches))
            for i, ms in enumerate(masks_splitting_cells_test_rand_patches):
                if len(np.where(ms == 127)[0]) > 0:
                    splitting_cells_test[i] = 2

        # 7) Data augmentation
        print('Data augmentation')
        if path_mask_splitting_cells:
            X_train_aug, y_train_aug, weight_maps_train_aug, splitting_cells_train_aug = self.data_augmentation(
                X_train_rand_patches, y_train_rand_patches, weight_maps_train_rand_patches, splitting_cells_train)
            X_val_aug, y_val_aug, weight_maps_val_aug, splitting_cells_val_aug = self.data_augmentation(
                X_val_rand_patches, y_val_rand_patches, weight_maps_val_rand_patches, splitting_cells_val)
            X_test_aug, y_test_aug, weight_maps_test_aug, splitting_cells_test_aug = self.data_augmentation(
                X_test_rand_patches, y_test_rand_patches, weight_maps_test_rand_patches, splitting_cells_test)
        else:
            X_train_aug, y_train_aug, weight_maps_train_aug = self.data_augmentation(
                X_train_rand_patches, y_train_rand_patches, weight_maps_train_rand_patches)
            X_val_aug, y_val_aug, weight_maps_val_aug = self.data_augmentation(
                X_val_rand_patches, y_val_rand_patches, weight_maps_val_rand_patches)
            X_test_aug, y_test_aug, weight_maps_test_aug = self.data_augmentation(
                X_test_rand_patches, y_test_rand_patches, weight_maps_test_rand_patches)

        # 8) Generate binary label for coverage of colored pixels
        #    (1 - < 0.75, 2 - > 0.75)
        print('Generate label for cell coverage')
        if path_mask_splitting_cells:
            ratio_cell_pixels_train = y_train_aug.sum(
                axis=(1, 2, 3)) / reduce(operator.mul, y_train_rand_patches.shape[1:], 1)
            ratio_cell_pixels_train_norm = ratio_cell_pixels_train / \
                                           ratio_cell_pixels_train.max()
            ratio_cell_pixels_train_bin = np.ones(
                ratio_cell_pixels_train_norm.shape)
            ratio_cell_pixels_train_bin[ratio_cell_pixels_train_norm > 0.75] = 2

            ratio_cell_pixels_val = y_val_aug.sum(
                axis=(1, 2, 3)) / reduce(operator.mul, y_val_rand_patches.shape[1:], 1)
            ratio_cell_pixels_val_norm = ratio_cell_pixels_val / ratio_cell_pixels_val.max()
            ratio_cell_pixels_val_bin = np.ones(
                ratio_cell_pixels_val_norm.shape)
            ratio_cell_pixels_val_bin[ratio_cell_pixels_val_norm > 0.75] = 2

            ratio_cell_pixels_test = y_test_aug.sum(
                axis=(1, 2, 3)) / reduce(operator.mul, y_test_rand_patches.shape[1:], 1)
            ratio_cell_pixels_test_norm = ratio_cell_pixels_test / ratio_cell_pixels_test.max()
            ratio_cell_pixels_test_bin = np.ones(
                ratio_cell_pixels_test_norm.shape)
            ratio_cell_pixels_test_bin[ratio_cell_pixels_test_norm > 0.75] = 2
        else:
            ratio_cell_pixels_train = y_train_aug.sum(
                axis=(1, 2, 3)) / reduce(operator.mul, y_train_rand_patches.shape[1:], 1)
            ratio_cell_pixels_train_norm = ratio_cell_pixels_train / \
                                           ratio_cell_pixels_train.max()
            ratio_cell_pixels_train_bin = np.ones(
                ratio_cell_pixels_train_norm.shape)
            ratio_cell_pixels_train_bin[ratio_cell_pixels_train_norm > 0.75] = 2

            ratio_cell_pixels_val = y_val_aug.sum(
                axis=(1, 2, 3)) / reduce(operator.mul, y_val_rand_patches.shape[1:], 1)
            ratio_cell_pixels_val_norm = ratio_cell_pixels_val / ratio_cell_pixels_val.max()
            ratio_cell_pixels_val_bin = np.ones(
                ratio_cell_pixels_val_norm.shape)
            ratio_cell_pixels_val_bin[ratio_cell_pixels_val_norm > 0.75] = 2

            ratio_cell_pixels_test = y_test_aug.sum(
                axis=(1, 2, 3)) / reduce(operator.mul, y_test_rand_patches.shape[1:], 1)
            ratio_cell_pixels_test_norm = ratio_cell_pixels_test / ratio_cell_pixels_test.max()
            ratio_cell_pixels_test_bin = np.ones(
                ratio_cell_pixels_test_norm.shape)
            ratio_cell_pixels_test_bin[ratio_cell_pixels_test_norm > 0.75] = 2

        # 9) Return values
        if path_mask_splitting_cells:
            return (
                X_train_aug,
                y_train_aug,
                weight_maps_train_aug,
                splitting_cells_train_aug,
                ratio_cell_pixels_train_bin,
                X_val_aug,
                y_val_aug,
                weight_maps_val_aug,
                splitting_cells_val_aug,
                ratio_cell_pixels_val_bin,
                X_test_aug,
                y_test_aug,
                weight_maps_test_aug,
                splitting_cells_test_aug,
                ratio_cell_pixels_test_bin)
        else:
            return (
                X_train_aug,
                y_train_aug,
                weight_maps_train_aug,
                ratio_cell_pixels_train_bin,
                X_val_aug,
                y_val_aug,
                weight_maps_val_aug,
                ratio_cell_pixels_val_bin,
                X_test_aug,
                y_test_aug,
                weight_maps_test_aug,
                ratio_cell_pixels_test_bin)

    def run_mother_machine(self, path_img, path_mask):
        '''
        Run the preprocessing for mother machine:
            1) Load data
            2) Pad images and masks to adjust height and width
            3) Split images and masks into train and validation set
            4) Split each image and mask into overlapping horizontal patches
            5) Data augmentation

            2) Generate weight map
            3) Split image, masks and weight map into 4x4-grid

            5) Split larger patches from 4x4-grid into
                smaller ones
            6) Extract binary weight for splitting events from mask
               (1 - no splitting event, 2 - more than one)
            7) Data augmentation
            8) Generate binary label for coverage of colored pixels
               (1 - < 0.75, 2 - > 0.75)
            9) Return values
        '''

        # 1) Load data
        list_imgs = np.sort(os.listdir(path_img))
        imgs_unpad = [self.scale_pixel_vals(
            io.imread(path_img + i, as_gray=True)) for i in list_imgs]
        list_masks = np.sort(os.listdir(path_mask))
        masks_unpad = [self.scale_pixel_vals(
            io.imread(path_mask + m).astype(int)) for m in list_masks]

        # 2) Pad images and masks to adjust height and width
        self.get_max_shape(imgs_unpad)
        imgs = np.array([self.pad_image(i) for i in imgs_unpad])
        masks = np.array([self.pad_image(m) for m in masks_unpad])

        # 3) Split images and masks into train and validation set
        X_train, X_val, y_train, y_val = train_test_split(
            imgs, masks, test_size=self.test_size)

        # 4) Split each image and mask into horizontal patches
        self.define_horizontal_splits(imgs[0])
        X_train_patch = np.concatenate(
            [self.horizontal_split(xtrain) for xtrain in X_train])
        X_val_patch = np.concatenate(
            [self.horizontal_split(xval) for xval in X_val])
        y_train_patch = np.concatenate(
            [self.horizontal_split(ytrain) for ytrain in y_train])
        y_val_patch = np.concatenate(
            [self.horizontal_split(yval) for yval in y_val])

        # 5) Data augmentation
        # currently the generation of weight maps for this dataset type
        # (mother machine) is not working. For this reason y_train_patch
        # and y_val_patch are handed to the method for the data augmentation
        # as a dummy instead of the weight maps.
        X_train_aug, y_train_aug, weight_maps_train_aug = \
            self.data_augmentation(X_train_patch, y_train_patch, y_train_patch)
        X_val_aug, y_val_aug, weight_maps_val_aug = \
            self.data_augmentation(X_val_patch, y_val_patch, y_val_patch)

        return X_train_aug, y_train_aug, X_val_aug, y_val_aug

    def generate_patches(self, img, mask, weight_map,
                         mask_splitting_event=None):
        '''
        Split image, mask and weight map into a 4x4-grid.
        '''

        # define height and width of patches
        num_patches = 4
        shift_r = int(img.shape[0] / (num_patches * 2))
        shift_c = int(img.shape[1] / (num_patches * 2))

        # define coordinates of patches
        rows = np.arange(shift_r, img.shape[0], shift_r * 2).astype(int)
        cols = np.arange(shift_c, img.shape[1], shift_c * 2).astype(int)
        rows, cols = np.meshgrid(rows, cols)
        rows = rows.flatten()
        cols = cols.flatten()

        # split imgs, masks and weight maps into patches
        imgs = []
        masks = []
        weight_maps = []
        if mask_splitting_event is not None:
            masks_splitting_events = []
        for r, c in zip(rows, cols):
            imgs.append(img[r - shift_r:r + shift_r, c - shift_c:c +
                                                                 shift_c].reshape((shift_r * 2, shift_c * 2, 1)))
            mask_patch = mask[r - shift_r:r + shift_r, c - shift_c:c +
                                                                   shift_c].reshape((shift_r * 2, shift_c * 2, 1))
            masks.append(mask_patch)
            weight_maps.append(weight_map[r -
                                          shift_r:r +
                                                  shift_r, c -
                                                           shift_c:c +
                                                                   shift_c].reshape((shift_r *
                                                                                     2, shift_c *
                                                                                     2, 1)))
            if mask_splitting_event is not None:
                masks_splitting_events.append(
                    mask_splitting_event[r - shift_r:r + shift_r, c - shift_c:c + shift_c].reshape(
                        (shift_r * 2, shift_c * 2, 1)))
        imgs = np.array(imgs)
        masks = np.array(masks)
        weight_maps = np.array(weight_maps)
        if mask_splitting_event is not None:
            masks_splitting_events = np.array(masks_splitting_events)

        if mask_splitting_event is not None:
            return imgs, masks, weight_maps, masks_splitting_events
        else:
            return imgs, masks, weight_maps

    def generate_random_patches(
            self, img, mask, weight_map, mask_splitting_event=None):
        '''
        Generate smaller patches from single patch of 4x4-grid.
        The smaller patches have a size of 128x128 pixels.
        '''

        # define coordinates of patches
        shift_r = 64
        shift_c = 64
        rows = np.linspace(shift_r, img.shape[0] - shift_r, self.num_split_r).astype(int)
        cols = np.linspace(shift_c, img.shape[1] - shift_c, self.num_split_c).astype(int)

        rows, cols = np.meshgrid(rows, cols)
        rows = rows.flatten()
        cols = cols.flatten()

        # split imgs, masks and weight maps into patches
        imgs = []
        masks = []
        weight_maps = []
        if mask_splitting_event is not None:
            masks_splitting_events = []
        for r, c in zip(rows, cols):
            imgs.append(img[r - shift_r:r + shift_r, c - shift_c:c +
                                                                 shift_c].reshape((shift_r * 2, shift_c * 2, 1)))
            mask_patch = mask[r - shift_r:r + shift_r, c - shift_c:c +
                                                                   shift_c].reshape((shift_r * 2, shift_c * 2, 1))
            masks.append(mask_patch)
            weight_maps.append(weight_map[r -
                                          shift_r:r +
                                                  shift_r, c -
                                                           shift_c:c +
                                                                   shift_c].reshape((shift_r *
                                                                                     2, shift_c *
                                                                                     2, 1)))
            if mask_splitting_event is not None:
                masks_splitting_events.append(
                    mask_splitting_event[r - shift_r:r + shift_r, c - shift_c:c + shift_c].reshape(
                        (shift_r * 2, shift_c * 2, 1)))

        imgs = np.array(imgs)
        masks = np.array(masks)
        weight_maps = np.array(weight_maps)
        if mask_splitting_event is not None:
            masks_splitting_events = np.array(masks_splitting_events)

        if mask_splitting_event is not None:
            return imgs, masks, weight_maps, masks_splitting_events
        else:
            return imgs, masks, weight_maps

    def scale_pixel_vals(self, img):
        '''
        Scale pivel values between 0 and 1.
        '''

        img = np.array(img)
        return ((img - img.min()) / (img.max() - img.min()))

    def generate_weight_map(self, mask):
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

    def compute_pixel_ratio(self, masks):
        '''
        Compute the ratio of colored pixels in a mask.
        '''

        return np.array(
            [int(np.round(100 * np.count_nonzero(m) / len(m.flatten()))) for m in masks])

    def stratified_split(self, test_size, X, y, weight_maps, ratio,
                         label_splitting_cells=None):
        '''
        Split data depending on the ratio of colored pixels in all images (patches).
        '''

        sss = StratifiedShuffleSplit(test_size=test_size)
        for train_idx, test_idx in sss.split(X, ratio):
            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            weight_maps_train = weight_maps[train_idx]
            weight_maps_test = weight_maps[test_idx]

            ratio_train = ratio[train_idx]
            ratio_test = ratio[test_idx]

            if label_splitting_cells:
                label_splitting_cells_train = label_splitting_cells[train_idx]
                label_splitting_cells_test = label_splitting_cells[test_idx]

        if label_splitting_cells:
            return X_train, X_test, y_train, y_test, weight_maps_train, \
                   weights_maps_test, ratio_train, ratio_test, \
                   label_splitting_cells_train, label_splitting_cells_test

        else:
            return X_train, X_test, y_train, y_test, weight_maps_train, \
                   weights_maps_test, ratio_train, ratio_test

    def split_data(self, imgs, masks, weight_maps, ratio,
                   label_splitting_cells=None):
        '''
        Split data depending on the ratio of colored pixels in all images (patches).
        '''

        try:
            if label_splitting_cells:
                (X_train,
                 X_test,
                 y_train,
                 y_test,
                 weight_maps_train,
                 weights_maps_test,
                 ratio_train,
                 ratio_test,
                 label_splitting_cells_train,
                 label_splitting_cells_test) = self.stratified_split(self.test_size, imgs, masks, weight_maps, ratio,
                                                                     label_splitting_cells)

                (X_train,
                 X_val,
                 y_train,
                 y_val,
                 weight_maps_train,
                 weights_maps_val,
                 ratio_train,
                 ratio_val,
                 label_splitting_cells_train,
                 label_splitting_cells_val) = self.stratified_split(self.val_size, X_train, y_train, weight_maps_train,
                                                                    ratio_train, label_splitting_cells_train)

            else:
                (X_train,
                 X_test,
                 y_train,
                 y_test,
                 weight_maps_train,
                 weights_maps_test,
                 ratio_train,
                 ratio_test) = self.stratified_split(self.test_size, imgs, masks, weight_maps, ratio)

                (X_train,
                 X_val,
                 y_train,
                 y_val,
                 weight_maps_train,
                 weights_maps_val,
                 ratio_train,
                 ratio_val) = self.stratified_split(self.val_size, X_train, y_train, weight_maps_train, ratio_train)

        # sss = StratifiedShuffleSplit(test_size=self.test_size)
        # try:
        #    for train_idx, val_idx in sss.split(imgs, ratio):
        #        X_train = imgs[train_idx]
        #        X_val = imgs[val_idx]

        #        y_train = masks[train_idx]
        #        y_val = masks[val_idx]

        #        weight_maps_train = weight_maps[train_idx]
        #        weight_maps_val = weight_maps[val_idx]

        #        if label_splitting_cells:
        #            label_splitting_cells_train = label_splitting_cells[train_idx]
        #            label_splitting_cells_val = label_splitting_cells[val_idx]

        # In case a ValueError occurs and an even split among
        # the classes is not possible, standard train-test split is applied.
        except ValueError:
            if label_splitting_cells is not None:

                (X_train,
                 X_test,
                 y_train,
                 y_test,
                 weight_maps_train,
                 weight_maps_test,
                 label_splitting_cells_train,
                 label_splitting_cells_test) = train_test_split(imgs,
                                                                masks,
                                                                weight_maps,
                                                                label_splitting_cells,
                                                                test_size=self.test_size)
                (X_train,
                 X_val,
                 y_train,
                 y_val,
                 weight_maps_train,
                 weight_maps_val,
                 label_splitting_cells_train,
                 label_splitting_cells_val) = train_test_split(X_train,
                                                               y_train,
                                                               weight_maps_train,
                                                               label_splitting_cells_train,
                                                               test_size=self.val_size)
            else:
                X_train, X_test, y_train, y_test, weight_maps_train, weight_maps_test = \
                    train_test_split(imgs, masks, weight_maps,
                                     test_size=self.test_size)

                X_train, X_val, y_train, y_val, weight_maps_train, weight_maps_val = \
                    train_test_split(X_train, y_train, weight_maps_train,
                                     test_size=self.val_size)

        if label_splitting_cells is not None:
            return X_train, y_train, weight_maps_train, label_splitting_cells_train, \
                   X_val, y_val, weight_maps_val, label_splitting_cells_val, \
                   X_test, y_test, weight_maps_test, label_splitting_cells_test
        else:
            return X_train, y_train, weight_maps_train, X_val, y_val, weight_maps_val, \
                   X_test, y_test, weight_maps_test

    def define_horizontal_splits(self, img):
        '''
        Define the starting heights of the horizontal splits.
        The height of the resulting horizontal patch is quarter the height of the
        original image. The width of the horizontal patch is the width of the original
        image.
        '''

        height, _ = img.shape

        # define the minimum height (split_start) of random splits
        num_splits = 150
        # random patch should have the quarter height of the original image
        self.height_cutoff = 16 * (height // (4 * 16))
        self.split_start = np.linspace(0, height, num_splits, dtype=int)

    def horizontal_split(self, img):
        '''
        Split an image into random horizontal patches accoring to
        previously defined starting heights of the horizontal splits. The image is
        horizontally mirrored before the split to to use the full height of the image.
        '''

        # split image according to where split starts
        hor_splits = []
        img_mirrored = self.pad_horizontally(img)
        for start in self.split_start:
            hor_splits.append(
                img_mirrored[start:(start + self.height_cutoff), :])
        hor_splits = np.array(hor_splits)

        return hor_splits

    def get_max_shape(self, imgs):
        '''
        Get the maximal height and width across all images
        in a dataset. The maximal height and width are also set to a
        multiple of 16 to allow an even split of feature maps in the unet.
        '''

        imgs_shape = [i.shape for i in imgs]
        self.height_max = int(
            np.ceil(np.max([s[0] for s in imgs_shape]) / 16) * 16)
        self.width_max = int(
            np.ceil(np.max([s[1] for s in imgs_shape]) / 16) * 16)

    def pad_horizontally(self, img):
        '''
        Mirror an image horizontally.
        '''

        horizontal_mirrored = np.concatenate([img, img[::-1, :]])
        return horizontal_mirrored

    def pad_image(self, img):
        '''
        Pad an image with its horizontal and vertical mirror
        image and reduces the image size according to the maximal height
        and width of all images in the dataset.
        '''

        horizontal_mirrored = np.concatenate([img, img[::-1, :]])
        horizontal_pad = horizontal_mirrored[:self.height_max, :]
        vertical_mirrored = np.hstack(
            [horizontal_pad, horizontal_pad[:, ::-1]])
        final_pad = vertical_mirrored[:, :self.width_max]

        return final_pad

    def data_augmentation(self, imgs, masks, weight_maps,
                          splitting_cells=None):
        '''
        Augments list of imgs, masks and weights maps by vertical and
        horizontal flips and increase and decrease of brightness.
        '''

        # vertical flip
        imgs_vert_flip = np.array([np.flipud(i) for i in imgs])
        masks_vert_flip = np.array([np.flipud(m) for m in masks])
        weight_maps_vert_flip = np.array([np.flipud(wm) for wm in weight_maps])

        # horizontal flip
        imgs_hor_flip = np.array([np.fliplr(i) for i in imgs])
        masks_hor_flip = np.array([np.fliplr(m) for m in masks])
        weight_maps_hor_flip = np.array([np.fliplr(wm) for wm in weight_maps])

        # decrease brightness
        imgs_decreased_brightness1 = []
        for i in imgs:
            imgs_decreased_brightness1.append(
                exposure.adjust_gamma(i, gamma=1.2, gain=1))
        imgs_decreased_brightness1 = np.array(imgs_decreased_brightness1)

        imgs_decreased_brightness2 = []
        for i in imgs:
            imgs_decreased_brightness2.append(
                exposure.adjust_gamma(i, gamma=1.4, gain=1))
        imgs_decreased_brightness2 = np.array(imgs_decreased_brightness2)

        imgs_decreased_brightness3 = []
        for i in imgs:
            imgs_decreased_brightness3.append(
                exposure.adjust_gamma(i, gamma=1.6, gain=1))
        imgs_decreased_brightness3 = np.array(imgs_decreased_brightness3)

        # enhance brightness
        imgs_enhanced_brightness1 = []
        for i in imgs:
            imgs_enhanced_brightness1.append(
                exposure.adjust_gamma(i, gamma=0.4, gain=1))
        imgs_enhanced_brightness1 = np.array(imgs_enhanced_brightness1)

        imgs_enhanced_brightness2 = []
        for i in imgs:
            imgs_enhanced_brightness2.append(
                exposure.adjust_gamma(i, gamma=0.4, gain=1))
        imgs_enhanced_brightness2 = np.array(imgs_enhanced_brightness2)

        imgs_enhanced_brightness3 = []
        for i in imgs:
            imgs_enhanced_brightness3.append(
                exposure.adjust_gamma(i, gamma=0.4, gain=1))
        imgs_enhanced_brightness3 = np.array(imgs_enhanced_brightness3)

        # combine all augmented patches of the images
        all_imgs = np.concatenate(
            (imgs,
             imgs_vert_flip,
             imgs_hor_flip,
             imgs_decreased_brightness1,
             imgs_decreased_brightness2,
             imgs_decreased_brightness3,
             imgs_enhanced_brightness1,
             imgs_enhanced_brightness2,
             imgs_enhanced_brightness3))

        # combine all augmented patches of the masks
        # for those images with increased or decreased brightness
        # the mask does not change since this modification of the image does not
        # affect the mask
        all_masks = np.concatenate(
            (masks,
             masks_vert_flip,
             masks_hor_flip,
             masks,
             masks,
             masks,
             masks,
             masks,
             masks))
        all_weight_maps = np.concatenate(
            (weight_maps,
             weight_maps_vert_flip,
             weight_maps_hor_flip,
             weight_maps,
             weight_maps,
             weight_maps,
             weight_maps,
             weight_maps,
             weight_maps))

        # in case an array for cell splitting events is provided the augmented
        # array corresponds to the original one as the number of splitting
        # events does not change with the data augmentation
        if splitting_cells is not None:
            all_splitting_cells = np.concatenate(
                (splitting_cells,
                 splitting_cells,
                 splitting_cells,
                 splitting_cells,
                 splitting_cells,
                 splitting_cells,
                 splitting_cells,
                 splitting_cells,
                 splitting_cells))

        # shuffle data and return augmented data either for cases containing
        # an array for cell splitting events or not
        shuffle_index = np.random.permutation(all_imgs.shape[0])
        if splitting_cells is not None:
            all_imgs, all_masks, all_weight_maps, all_splitting_cells = \
                all_imgs[shuffle_index], all_masks[shuffle_index], \
                all_weight_maps[shuffle_index], all_splitting_cells[shuffle_index]
            return all_imgs, all_masks, all_weight_maps, all_splitting_cells
        else:
            all_imgs, all_masks, all_weight_maps = \
                all_imgs[shuffle_index], all_masks[shuffle_index], all_weight_maps[shuffle_index]
            return all_imgs, all_masks, all_weight_maps
