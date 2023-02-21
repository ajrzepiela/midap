import os
from configparser import ConfigParser
from pathlib import Path
from typing import Optional, List, Union, Tuple
from skimage import io

import numpy as np
import tensorflow as tf

from .preprocessing import DataProcessor


class TFPipe(DataProcessor):
    """
    This class can be used to createa datasets for TF model training
    """

    def __init__(self, paths: Union[str, bytes, os.PathLike, List[Union[str, bytes, os.PathLike]]],
                 n_grid=4, test_size=0.15, val_size=0.2, sigma=2.0, w_0=2.0, w_c0=1.0, w_c1=1.1, loglevel=7,
                 np_random_seed: Optional[int] = None, batch_size=32, shuffle_buffer=128, image_size=(128, 128, 1),
                 delta_brightness: Optional[float] = 0.4, lower_contrast: Optional[float] = 0.2,
                 upper_contrast: Optional[float] = 0.5, n_repeats: Optional[int] = 50,
                 train_seed: Optional[tuple] = None, val_seed=(11, 12), test_seed=(13, 14)):
        """
        Initializes the TFPipe instance. Note that a lot parameters are used to implement the weight map
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
        :param batch_size: The batch size of the data sets
        :param shuffle_buffer: The shuffle buffer used for the training set
        :param image_size: The target image size including channel dimension
        :param delta_brightness: The max delta_brightness for random brightness adjustments,
                                 can be None -> no adjustments
        :param lower_contrast: The lower limit for random contrast adjustments, can be None -> no adjustments
        :param upper_contrast: The upper limit for random contrast adjustments, can be None -> no adjustments
        :param n_repeats: The number of repeats of random operations per original image, i.e. number of data
                          augmentations
        :param train_seed: A tuple of two seed used to seed the stateless random operations of the training dataset.
                           If set to None (default) each iteration through the training set will have different
                           random augmentations, if set the same augmentations will be used every iteration. Note that
                           even if this seed is set, the shuffling operation will still be truly random if the
                           shuffle_buffer > 1
        :param val_seed: The seed for the validation set (see train_seed), defaults to (11, 12) for reproducibility
        :param test_seed: The seed for the test set (see train_seed), defaults to (13, 14) for reproducibility
        """

        # we make a copy of the locals here to make the config gen less painful
        locals_copy = locals()

        # init the base class
        super().__init__(paths=paths, n_grid=n_grid, test_size=test_size, val_size=val_size, sigma=sigma,
                         w_0=w_0, w_c0=w_c0, w_c1=w_c1, loglevel=loglevel, np_random_seed=np_random_seed)
        
        # get the datasets
        self.data_dict = self.get_dset()

        # check the size
        shapes = [x.shape[1:] for x in self.data_dict["X_train"]]
        min_shape = np.min(np.array(shapes), axis=0)
        for i in range(3):
            if min_shape[i] < image_size[i]:
                raise ValueError(f"The image_size of {image_size} is not compatible with the training data, "
                                 f"max possible shape is {tuple(min_shape)}!")

        # set the TF datasets
        self.set_tf_dsets(batch_size=batch_size, shuffle_buffer=shuffle_buffer, image_size=image_size,
                          delta_brightness=delta_brightness, lower_contrast=lower_contrast,
                          upper_contrast=upper_contrast, n_repeats=n_repeats, train_seed=train_seed, val_seed=val_seed,
                          test_seed=test_seed)

        # create the config for the meta data
        self.config = ConfigParser()
        self.config.add_section("TFPipe")
        self.config.set("TFPipe", "train_files", " \n".join([f"{p}" for p in self.img_paths]))
        for key, val in locals_copy.items():
            # Fun fact: python adds the __class__ key to the locals because "super" is mentioned in this function
            if key not in ["paths", "self", "__class__"]:
                self.config.set("TFPipe", key, f"{val}")


    @staticmethod
    def _map_crop(num: tf.Tensor, imgs: Tuple[tf.Tensor], target_size: tuple,
                  stateless_seed: Optional[tuple]=None):
        """
        Performs a crop operation on image, weight and label map
        :param num: A Tensor with the number of the sample used to increase the stateless seed (if provided)
        :param imgs: A suple of the image, weight and label map
        :param target_size: The target size (with channel dim)
        :param stateless_seed: If provided, used as input for the stateless crop operation, otherwise and unseeded
                               stateful crop is used
        :return: The tensors of value, cropped
        """

        # combine
        i, w, l = imgs
        stack = tf.stack([i, w, l], axis=-1)

        if stateless_seed is None:
            out = tf.image.random_crop(value=stack, size=target_size + (3, ))
        else:
            seed = tf.convert_to_tensor(stateless_seed, dtype=tf.int32) + tf.cast(num, dtype=tf.int32)
            out = tf.image.stateless_random_crop(value=stack, size=target_size + (3,), seed=seed)

        return num, tuple(out[...,i] for i in range(3))

    @staticmethod
    def _map_brightness(num: tf.Tensor, imgs: Tuple[tf.Tensor], max_delta: float,
                        stateless_seed: Optional[tuple]=None):
        """
        Performs a brightness adjust operation on image, leaves weight and label map
        :param num: A Tensor with the number of the sample used to increase the stateless seed (if provided)
        :param imgs: A suple of the image, weight and label map
        :param max_delta: The maximum delta to adjust the brightness
        :param stateless_seed: If provided, used as input for the stateless crop operation, otherwise and unseeded
                               stateful crop is used
        :return: The tensors of value, cropped
        """

        i, w, l = imgs
        if stateless_seed is None:
            i = tf.image.random_brightness(image=i, max_delta=max_delta)
        else:
            seed = tf.convert_to_tensor(stateless_seed, dtype=tf.int32) + tf.cast(num, dtype=tf.int32)
            i = tf.image.stateless_random_brightness(image=i, max_delta=max_delta, seed=seed)

        return num, (i, w, l)

    @staticmethod
    def _map_contrast(num: tf.Tensor, imgs: Tuple[tf.Tensor], lower: float, upper: float,
                      stateless_seed: Optional[tuple]=None):
        """
        Performs a contrast adjust operation on image, leaves weight and label map
        :param num: A Tensor with the number of the sample used to increase the stateless seed (if provided)
        :param imgs: A suple of the image, weight and label map
        :param lower: The lower bound for the contrast adjust
        :param upper: The upper bound for the contrast adjust
        :param stateless_seed: If provided, used as input for the stateless crop operation, otherwise and unseeded
                               stateful crop is used
        :return: The tensors of value, cropped
        """

        i, w, l = imgs
        if stateless_seed is None:
            i = tf.image.random_contrast(image=i, lower=lower, upper=upper)
        else:
            seed = tf.convert_to_tensor(stateless_seed, dtype=tf.int32) + tf.cast(num, dtype=tf.int32)
            i = tf.image.stateless_random_contrast(image=i, lower=lower, upper=upper, seed=seed)

        return num, (i, w, l)

    @staticmethod
    def _map_ud_flip(num: tf.Tensor, imgs: Tuple[tf.Tensor], stateless_seed: Optional[tuple] = None):
        """
        Performs a random flip along the first dimension (up down)
        :param num: A Tensor with the number of the sample used to increase the stateless seed (if provided)
        :param imgs: A suple of the image, weight and label map
        :param stateless_seed: If provided, used as input for the stateless crop operation, otherwise and unseeded
                               stateful crop is used
        :return: The tensors of value, cropped
        """

        # combine
        i, w, l = imgs
        stack = tf.stack([i, w, l], axis=0)

        if stateless_seed is None:
            out = tf.image.random_flip_up_down(image=stack)
        else:
            seed = tf.convert_to_tensor(stateless_seed, dtype=tf.int32) + tf.cast(num, dtype=tf.int32)
            out = tf.image.stateless_random_flip_up_down(image=stack, seed=seed)

        return num, tuple(out[i, ...] for i in range(3))

    @staticmethod
    def _map_lr_flip(num: tf.Tensor, imgs: Tuple[tf.Tensor], stateless_seed: Optional[tuple] = None):
        """
        Performs a random flip along the second dimension (left right)
        :param num: A Tensor with the number of the sample used to increase the stateless seed (if provided)
        :param imgs: A suple of the image, weight and label map
        :param stateless_seed: If provided, used as input for the stateless crop operation, otherwise and unseeded
                               stateful crop is used
        :return: The tensors of value, cropped
        """

        # combine
        i, w, l = imgs
        stack = tf.stack([i, w, l], axis=0)

        if stateless_seed is None:
            out = tf.image.random_flip_left_right(image=stack)
        else:
            seed = tf.convert_to_tensor(stateless_seed, dtype=tf.int32) + tf.cast(num, dtype=tf.int32)
            out = tf.image.stateless_random_flip_left_right(image=stack, seed=seed)

        return num, tuple(out[i, ...] for i in range(3))

    @staticmethod
    def zip_inputs(images: np.ndarray, weights: np.ndarray, segmentations: np.ndarray):
        """
        Puts images, weights and segmentations (labels) into a TF dataset
        :param images: The BWHC input of the images
        :param weights: The BWHC input of the weights
        :param segmentations: The BWHC input of the labels
        :return: A single TF dataset containing the elements in that sequence
        """

        img_dset = tf.data.Dataset.from_tensor_slices(images.astype(np.float32))
        w_dset = tf.data.Dataset.from_tensor_slices(weights.astype(np.float32))
        seg_dset = tf.data.Dataset.from_tensor_slices(segmentations.astype(np.float32))

        return tf.data.Dataset.zip((img_dset, w_dset, seg_dset))

    def set_tf_dsets(self, batch_size, shuffle_buffer=128, image_size=(128, 128, 1), delta_brightness: Optional[float] = 0.4,
                     lower_contrast: Optional[float] = 0.2, upper_contrast: Optional[float] = 0.5,
                     n_repeats: Optional[int] = 50, train_seed: Optional[tuple] = None, val_seed=(11, 12),
                     test_seed=(13, 14)):
        """
        Creates train, test and validation TF datasets.
        :param batch_size: The batch size of the data sets
        :param shuffle_buffer: The shuffle buffer used for the training set
        :param image_size: The target image size including channel dimension
        :param delta_brightness: The max delta_brightness for random brightness adjustments, 
                                 can be None -> no adjustments
        :param lower_contrast: The lower limit for random contrast adjustments, can be None -> no adjustments
        :param upper_contrast: The upper limit for random contrast adjustments, can be None -> no adjustments
        :param n_repeats: The number of repeats of random operations per original image, i.e. number of data 
                          augmentations 
        :param train_seed: A tuple of two seed used to seed the stateless random operations of the training dataset. 
                           If set to None (default) each iteration through the training set will have different 
                           random augmentations, if set the same augmentations will be used every iteration. Note that 
                           even if this seed is set, the shuffling operation will still be truly random if the 
                           shuffle_buffer > 1
        :param val_seed: The seed for the validation set (see train_seed), defaults to (11, 12) for reproducibility
        :param test_seed: The seed for the test set (see train_seed), defaults to (13, 14) for reproducibility 
        """

        # stack imgs, weights and labels together
        self.dsets_train = [self.zip_inputs(i, w, l) for i, w, l in zip(self.data_dict["X_train"],
                                                                        self.data_dict["weight_maps_train"],
                                                                        self.data_dict["y_train"])]
        self.dsets_test = [self.zip_inputs(i, w, l) for i, w, l in zip(self.data_dict["X_test"],
                                                                       self.data_dict["weight_maps_test"],
                                                                       self.data_dict["y_test"])]
        self.dsets_val = [self.zip_inputs(i, w, l) for i, w, l in zip(self.data_dict["X_val"],
                                                                      self.data_dict["weight_maps_val"],
                                                                      self.data_dict["y_val"])]

        # now we repeat each dataset, such that we can have multple different crops etc.
        self.dsets_train = [d.repeat(n_repeats).enumerate() for d in self.dsets_train]
        self.dsets_test = [d.repeat(n_repeats).enumerate() for d in self.dsets_test]
        self.dsets_val = [d.repeat(n_repeats).enumerate() for d in self.dsets_val]

        # crop
        self.dsets_train = [
            d.map(lambda num, imgs: self._map_crop(num, imgs, target_size=image_size, stateless_seed=train_seed))
            for d in self.dsets_train]
        self.dsets_test = [
            d.map(lambda num, imgs: self._map_crop(num, imgs, target_size=image_size, stateless_seed=test_seed))
            for d in self.dsets_test]
        self.dsets_val = [
            d.map(lambda num, imgs: self._map_crop(num, imgs, target_size=image_size, stateless_seed=val_seed))
            for d in self.dsets_val]
        # up and lr flips
        self.dsets_train = [
            d.map(lambda num, imgs: self._map_ud_flip(num, imgs, stateless_seed=train_seed))
            for d in self.dsets_train]
        self.dsets_test = [
            d.map(lambda num, imgs: self._map_ud_flip(num, imgs, stateless_seed=test_seed))
            for d in self.dsets_test]
        self.dsets_val = [
            d.map(lambda num, imgs: self._map_ud_flip(num, imgs, stateless_seed=val_seed))
            for d in self.dsets_val]
        self.dsets_train = [
            d.map(lambda num, imgs: self._map_lr_flip(num, imgs, stateless_seed=train_seed))
            for d in self.dsets_train]
        self.dsets_test = [
            d.map(lambda num, imgs: self._map_lr_flip(num, imgs, stateless_seed=test_seed))
            for d in self.dsets_test]
        self.dsets_val = [
            d.map(lambda num, imgs: self._map_lr_flip(num, imgs, stateless_seed=val_seed))
            for d in self.dsets_val]
        # perform the augmentations
        if delta_brightness is not None:
            self.dsets_train = [d.map(
                lambda num, imgs: self._map_brightness(num, imgs, max_delta=delta_brightness, stateless_seed=train_seed))
                for d in self.dsets_train]
            self.dsets_test = [d.map(
                lambda num, imgs: self._map_brightness(num, imgs, max_delta=delta_brightness, stateless_seed=test_seed))
                for d in self.dsets_test]
            self.dsets_val = [d.map(
                lambda num, imgs: self._map_brightness(num, imgs, max_delta=delta_brightness, stateless_seed=val_seed))
                for d in self.dsets_val]
        if lower_contrast is not None and upper_contrast is not None:
            self.dsets_train = [
                d.map(lambda num, imgs: self._map_contrast(num, imgs, lower=lower_contrast, upper=upper_contrast,
                                                         stateless_seed=train_seed))
                for d in self.dsets_train]
            self.dsets_test = [
                d.map(lambda num, imgs: self._map_contrast(num, imgs, lower=lower_contrast, upper=upper_contrast,
                                                         stateless_seed=test_seed))
                for d in self.dsets_test]
            self.dsets_val = [
                d.map(lambda num, imgs: self._map_contrast(num, imgs, lower=lower_contrast, upper=upper_contrast,
                                                         stateless_seed=val_seed))
                for d in self.dsets_val]

        # finalize train dset
        dset_train = tf.data.Dataset.sample_from_datasets(self.dsets_train)
        self.dset_train = dset_train.shuffle(shuffle_buffer)

        # test and vl are just concatenated
        self.dset_test = self.dsets_test[0]
        for d in self.dsets_test[1:]:
            self.dset_test.concatenate(d)
        self.dset_val = self.dsets_val[0]
        for d in self.dsets_val[1:]:
            self.dset_val.concatenate(d)

        # batch remap all the datasets such that they are compatible with the fit function
        self.dset_train = self.dset_train.batch(batch_size).map(lambda num, imgs: (imgs, imgs[2]))
        self.dset_test = self.dset_test.batch(batch_size).map(lambda num, imgs: (imgs, imgs[2]))
        self.dset_val = self.dset_val.batch(batch_size).map(lambda num, imgs: (imgs, imgs[2]))
