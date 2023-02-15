from pathlib import Path

import numpy as np
import pytest
import skimage.io as io
from pytest import mark

from midap.data import DataProcessor


# Tests
#######

@mark.usefixtures("tmpdir")
def test_init(tmpdir):
    """
    Tests the __init__ function of the DataProcessor class
    :param tmpdir: A fixture that sets up a tmp directory
    """

    # this throws a value error because the name is not in a valid format
    with pytest.raises(ValueError):
        DataProcessor(paths="wrong_format.tif")

    # now we create a file
    tmpdir.joinpath("right_format_raw.tif").touch()

    # this throws a non-exsist error because the seg file does not exist
    with pytest.raises(FileNotFoundError):
        DataProcessor(paths=["right_format_raw.tif"])

    # now we create a seg file
    tmpdir.joinpath("right_format_seg.tif").touch()

    # this throws a non-exsist error because the seg file does not exist
    with pytest.raises(FileNotFoundError):
        DataProcessor(paths=["right_format_raw.tif"])


def test_tile_img():
    """
    Tests the tile_img() function of the DataProcessor class
    """

    # set seed
    np.random.seed(11)

    # tile a 4x4 image into 1x1
    img_1 = np.random.normal(size=(4,4))
    tiles = DataProcessor.tile_img(img_1, n_grid=4, divisor=1)

    # check for shape
    assert tiles.shape == (16, 1, 1)
    # check for uniqueness
    assert np.unique(tiles.ravel()).size == img_1.ravel().size

    # This tiling is not possible
    with pytest.raises(AssertionError):
        tiles = DataProcessor.tile_img(img_1, n_grid=8, divisor=1)

    # a bit more difficult
    n, m = 128, 73
    img_2 = np.random.normal(size=(n, m))
    tiles = DataProcessor.tile_img(img_2, n_grid=8, divisor=2)

    # check for shape
    expected_x = n//8
    expected_y = (m//(2*8))*2
    assert tiles.shape == (8*8, expected_x, expected_y)


def test_scale_pixel_vals():
    """
    Tests the scale_pixel_vals() function of the DataProcessor class
    """

    # set seed
    np.random.seed(11)

    # generate a test image
    img = np.random.normal(size=(1345, 452))

    # scale the values
    scales_img = DataProcessor.scale_pixel_vals(img)

    # test for max and min
    assert np.isclose(scales_img.max(), 1.0)
    assert np.isclose(scales_img.min(), 0.0)


@mark.usefixtures("dir_setup")
def test_generate_weight_map(dir_setup):
    """
    Tests the generate_weight_map() function of the DataProcessor class
    :param dir_setup: a fixture that sets up a temp directory and returns its paths and paths that can be used to init
                      DataProcessor
    """

    # extract paths
    tmpdir, paths = dir_setup

    # get the instance
    sigma = 2.0
    w_0 = 2.0
    w_c0 = 1.0
    w_c1 = 1.1
    data_processor = DataProcessor(paths=paths, sigma=sigma, w_0=w_0, w_c0=w_c0, w_c1=w_c1)

    # create a test_mask
    test_mask = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0]])

    # get the weights
    weights = data_processor.generate_weight_map(test_mask)

    # check for the shape
    assert weights.shape == test_mask.shape
    # check that the cells have a weight of w_c1
    assert np.allclose(weights[test_mask == 1], w_c1)
    # far away empty pixels should have a weight very close to of w_c0
    assert np.isclose(weights[0,-1], w_c0, atol=0.001)
    # cells things that are very close to cells have a large weight
    expected = w_c0 + w_0*np.exp(-2**2/(2*sigma**2))
    assert np.isclose(weights[1, 0], expected)
    # this cell has two different distances
    expected = w_c0 + w_0 * np.exp(-3 ** 2 / (2 * sigma ** 2))
    assert np.isclose(weights[2, 1], expected)


def test_compute_pixel_ratio():
    """
    Tests the compute_pixel_ratio() function of the DataProcessor class
    """

    # create a test_mask
    n, m = 23, 11
    test_mask = np.zeros((2, n, m))
    # set random values to 1
    test_mask[0, 1, 2] = 1
    test_mask[0, 1, 3] = 1
    test_mask[0, 11, 4] = 1
    test_mask[0, 21, 5] = 1

    # calculate the ratio
    ratio = DataProcessor.compute_pixel_ratio(test_mask)
    # check the ratio shape
    assert ratio.shape == (2, )
    # check the ratio
    assert np.allclose(ratio, np.array([4.0/(n*m), 0.0]))


def test_get_quantile_classes():
    """
    Tests the get_quantile_classes() function of the DataProcessor class
    """

    # create an even array
    ratios_1 = np.linspace(0.0, 16.0, 8, endpoint=False)

    # create the class labels
    n = 4
    labels = DataProcessor.get_quantile_classes(x=ratios_1, n=n)

    # check for number of classes
    assert np.unique(labels).size == n
    # check for correct labels
    assert np.all(labels == np.array([0, 0, 1, 1, 2, 2, 3, 3]))

    # check unenven array
    ratios_1 = np.linspace(0.0, 16.0, 9, endpoint=False)

    # create the class labels
    n = 4
    labels = DataProcessor.get_quantile_classes(x=ratios_1, n=n)

    # check for number of classes
    assert np.unique(labels).size == n
    # check for correct labels
    assert np.all(labels == np.array([0, 0, 0, 1, 1, 2, 2, 3, 3]))


@mark.usefixtures("dir_setup")
def test_split_data(dir_setup):
    """
    Tests the split_data() function of the DataProcessor class
    :param dir_setup: a fixture that sets up a temp directory and returns its paths and paths that can be used to init
                      DataProcessor
    """

    # extract paths
    tmpdir, paths = dir_setup

    # get the instance
    test_size = 0.2
    val_size = 0.125
    data_processor = DataProcessor(paths=paths, test_size=test_size, val_size=val_size)

    # test data
    n = 16
    imgs = np.arange(n)[:,None,None]
    masks = np.arange(n)[:,None,None]
    weight_maps = np.arange(n)[:,None,None]

    # the ratios are sorted
    ratio = np.linspace(0.0, 1.0, n)

    # get the splits
    splits = data_processor.split_data(imgs=imgs, masks=masks, weight_maps=weight_maps, ratio=ratio)

    # check the number of entries
    assert len(splits) == 9

    # check the length of the sets
    expected_test = int(n*test_size) + 1
    assert len(splits["X_test"]) == expected_test
    expected_val = int((n - expected_test)*val_size) + 1
    assert len(splits["X_val"]) == expected_val
    assert len(splits["X_train"]) == n - expected_test - expected_val

    # checl that everything is there only once
    assert np.unique(np.concatenate([splits["X_train"], splits["X_val"], splits["X_test"]])).size == n


@mark.usefixtures("dir_setup")
def test_get_dset(dir_setup, monkeypatch):
    """
    Tests the get_dset function of the DataProcessor class
    :param dir_setup: a fixture that sets up a temp directory and returns its paths and paths that can be used to init
                      DataProcessor
    """

    # set seed
    np.random.seed(11)

    # extract paths
    tmpdir, paths = dir_setup

    # get the instance
    n_grid = 4
    test_size = 0.2
    val_size = 0.1
    data_processor = DataProcessor(paths=paths, n_grid=n_grid, test_size=test_size,
                                   val_size=val_size)

    # This is a helper function
    def data_loader(name: Path, *args, **kwargs):
        """
        A mock dataloader to test the run function
        :param name: The name of the dataset to load
        :param args: additional args (not used)
        :param kwargs: additional kwargs (not used)
        :return: The dataset
        """

        # fake data
        test_img = np.random.normal(size=(32, 64))
        test_mask = (test_img > 0.2).astype(int)

        if "raw" in name.name:
            return test_img
        if "seg" in name.name:
            return test_mask
        else:
            raise ValueError(f"Unkown name: {name}")

    # run the data_processor
    monkeypatch.setattr(io, 'imread', data_loader)
    data_out = data_processor.get_dset()

    # check the length
    assert len(data_out) == 9
    # check a random dimension, the dimension can calculated from the test set and n_grid etc
    # we get the test size
    n_test = int(n_grid*n_grid*test_size) + 1
    n_val = int((n_grid*n_grid - n_test)*val_size) + 1
    n_train = n_grid*n_grid - n_test - n_val
    # multiy with the random patch gen
    assert data_out["X_train"][0].shape[0] == n_train
