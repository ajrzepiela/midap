from typing import Callable

import numpy as np
import pytest
import tensorflow as tf
from pytest import mark

from midap.data.tf_pipeline import TFPipe


# Fixtures
##########

@pytest.fixture()
def sample_images():
    """
    A fixture that returns a image, weight and label map
    :return: i, w, l which are (512, 512, 1) tensors
    """

    # set the random seed and generate the images
    np.random.seed(11)
    i = np.random.normal(size=(512, 512, 1))
    w = np.random.normal(size=(512, 512, 1))
    l = np.random.normal(size=(512, 512, 1))

    return tf.convert_to_tensor(i), tf.convert_to_tensor(w), tf.convert_to_tensor(l)


# Tests
#######


def run_statics(static_method: Callable, i: tf.Tensor, w: tf.Tensor, l: tf.Tensor, image_only_transform: bool,**kwargs):
    """
    This test tests a static method of the TFPipe by running it stateless and stateful and check if the
    output are as expected
    :param static_method: The method to test
    :param i: The image (first argument of static method)
    :param w: The weight map (second argument of static method)
    :param l: The label (third argument of static method)
    :param image_only_transform: Whether all parts are transformed or the image only
    :param kwargs: Additional keyword arguments forwarded to the call (not the stateless seed!)
    """

    # we run in with state twice and compare, should be different
    # IMPORTANT NOTE: setting the global random state via tf.random.set_seed, will ruin this behaviour since the
    # random operations inside the static_methods are seeded in this case with a randomly generated seed, so
    # they produce the same output when the dataset it traversed multiple times. For pytest, this means that we should
    # not set a global seed anywhere, because otherwise the tests have a different behaviour depending on the order
    dset = TFPipe.zip_inputs(i.numpy()[None, ...], w.numpy()[None, ...], l.numpy()[None, ...]).enumerate()
    dset = dset.map(lambda num, imgs: static_method(num, imgs, stateless_seed=None, **kwargs))
    for num, (i1, w1, l1) in dset:
        pass
    for num, (i2, w2, l2) in dset:
        pass

    assert not np.allclose(i1.numpy(), i2.numpy())
    if image_only_transform:
        assert np.allclose(w1.numpy(), w2.numpy())
        assert np.allclose(l1.numpy(), l2.numpy())
    else:
        assert not np.allclose(w1.numpy(), w2.numpy())
        assert not np.allclose(l1.numpy(), l2.numpy())

    # we build it with state, each run through should be identical
    dset = TFPipe.zip_inputs(i.numpy()[None, ...], w.numpy()[None, ...], l.numpy()[None, ...]).enumerate()
    dset = dset.map(lambda num, imgs: static_method(num, imgs, stateless_seed=(11, 12), **kwargs))
    for num, (i1, w1, l1) in dset:
        pass
    for num, (i2, w2, l2) in dset:
        pass

    assert np.allclose(i1.numpy(), i2.numpy())
    assert np.allclose(w1.numpy(), w2.numpy())
    assert np.allclose(l1.numpy(), l2.numpy())


def test_zip_inputs(sample_images):
    """
    Tests the zip inputs statis
    :param sample_images: A fixture that creates sample images
    """

    # extract
    i, w, l = sample_images

    # we zip the inputs
    dset = TFPipe.zip_inputs(i.numpy()[None,...], w.numpy()[None,...], l.numpy()[None,...])

    for i1, w1, l1 in dset:
        assert np.allclose(i1.numpy(), i.numpy())
        assert np.allclose(w1.numpy(), w.numpy())
        assert np.allclose(l1.numpy(), l.numpy())


def test_map_crop(sample_images):
    """
    Tests the map crop method
    :param sample_images: A fixture that creates sample images
    """

    # extract
    i, w, l = sample_images

    # the cropping
    target_size = (128, 128, 1)
    run_statics(TFPipe._map_crop, i, w, l, image_only_transform=False, target_size=target_size)


def test_map_brightness(sample_images):
    """
    Tests the map brightness method
    :param sample_images: A fixture that creates sample images
    """

    # extract
    i, w, l = sample_images

    # the cropping
    max_delta = 0.4
    run_statics(TFPipe._map_brightness, i, w, l, image_only_transform=True, max_delta=max_delta)


def test_map_contrast(sample_images):
    """
    Tests the map contrast method
    :param sample_images: A fixture that creates sample images
    """

    # extract
    i, w, l = sample_images

    # the cropping
    lower = 0.2
    upper = 0.4
    run_statics(TFPipe._map_contrast, i, w, l, image_only_transform=True, lower=lower, upper=upper)


@mark.usefixtures("dir_setup")
def test_TFPipe(dir_setup):
    """
    Tests the init (and thus the run) of the TFPipe class
    :param dir_setup: A fixture that creates the necessary files and returns the tmpdir and paths to init the class
    """

    # unpack
    tmpdir, paths = dir_setup

    # this raises an error because the image size is too large
    with pytest.raises(ValueError):
        _ = TFPipe(paths=paths, batch_size=1, image_size=(128, 128, 1))

    # full run
    _ = TFPipe(paths=paths, batch_size=1, image_size=(32, 32, 1))
