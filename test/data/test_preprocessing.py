from midap.data import DataProcessor
import skimage.io as io
import numpy as np
import pytest
import os

# Tests
#######

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

def test_cut_patches():
    """
    Tests the cut_patches() function of the DataProcessor class
    """

    # set seed
    np.random.seed(11)

    # generate a test image
    img = np.random.normal(size=(1345, 452))

    # cut into some patches
    n_row = 4
    n_col = 3
    size = 74
    patches = DataProcessor.cut_patches(img=img, n_col=n_col, n_row=n_row, size=size)

    # check for shape
    assert patches.shape == (n_col*n_row, size, size)
    # check for uniqueness
    for patch in patches:
        assert (np.where(np.all(np.isclose(patch, patches), axis=(1, 2)))[0]).size == 1

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

def test_generate_weight_map():
    """
    Tests the generate_weight_map() function of the DataProcessor class
    """

    # set seed
    seed = 11

    # get the instance
    sigma = 2.0
    w_0 = 2.0
    w_c0 = 1.0
    w_c1 = 1.1
    data_processor = DataProcessor(np_random_seed=seed, sigma=sigma, w_0=w_0, w_c0=w_c0, w_c1=w_c1)

    # create a test_mask
    test_mask = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 0]])

    # get the weights
    weights = data_processor.generate_weight_map(test_mask)

    # check for the shape
    assert weights.shape == test_mask.shape + (1,)
    # check that the cells have a weight of w_c1
    assert np.allclose(weights[test_mask == 1], w_c1)
    # far away empty pixels should have a weight very close to of w_c0
    assert np.isclose(weights[0,-1, 0], w_c0, atol=0.001)
    # cells things that are very close to cells have a large weight
    expected = w_c0 + w_0*np.exp(-2**2/(2*sigma**2))
    assert np.isclose(weights[1, 0, 0], expected)
    # this cell has two different distances
    expected = w_c0 + w_0 * np.exp(-3 ** 2 / (2 * sigma ** 2))
    assert np.isclose(weights[2, 1, 0], expected)

def test_compute_pixel_ratio():
    """
    Tests the compute_pixel_ratio() function of the DataProcessor class
    """

    # set seed
    seed = 11

    # get the instance
    data_processor = DataProcessor(np_random_seed=seed)

    # create a test_mask
    n, m = 23, 11
    test_mask = np.zeros((2, n, m))
    # set random values to 1
    test_mask[0, 1, 2] = 1
    test_mask[0, 1, 3] = 1
    test_mask[0, 11, 4] = 1
    test_mask[0, 21, 5] = 1

    # calculate the ratio
    ratio = data_processor.compute_pixel_ratio(test_mask)
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

def test_split_data():
    """
    Tests the split_data() function of the DataProcessor class
    """

    # set seed
    seed = 11

    # get the instance
    test_size = 0.2
    val_size = 0.125
    data_processor = DataProcessor(np_random_seed=seed, test_size=test_size, val_size=val_size)

    # test data
    n = 16
    imgs = np.arange(n)[:,None,None]
    masks = np.arange(n)[:,None,None]
    weight_maps = np.arange(n)[:,None,None]
    label_splitting_cells = np.arange(n)[:,None,None]

    # the ratios are sorted
    ratio = np.linspace(0.0, 1.0, n)

    # get the splits
    splits = data_processor.split_data(imgs=imgs, masks=masks, weight_maps=weight_maps, ratio=ratio,
                                       label_splitting_cells=label_splitting_cells)

    # check the number of entries
    assert len(splits) == 12

    # check the length of the sets
    expected_test = int(n*test_size) + 1
    assert len(splits["X_test"]) == expected_test
    expected_val = int((n - expected_test)*val_size) + 1
    assert len(splits["X_val"]) == expected_val
    assert len(splits["X_train"]) == n - expected_test - expected_val

    # checl that everything is there only once
    assert np.unique(np.concatenate([splits["X_train"], splits["X_val"], splits["X_test"]])).size == n

def test_horizontal_split():
    """
    Tests the horizontal_split() function of the DataProcessor class
    """

    # set seed
    seed = 11

    # get the instance
    data_processor = DataProcessor(np_random_seed=seed)

    # create a test image
    n, m = 234, 16
    test_img = np.random.normal(size=(n, m))

    # get the lsplits
    height = 123
    num_splits = 11
    height_cutoff, split_start = data_processor.define_horizontal_splits(height=height, num_splits=num_splits)

    # the cutoff and the number of splits
    assert height_cutoff == height//4
    assert split_start.size == num_splits

    # now get the splits
    splits = data_processor.horizontal_split(img=test_img, height_cutoff=height_cutoff, split_start=split_start)

    # check for shape
    assert splits.shape == (num_splits, height_cutoff, m)
    # check for uniqueness
    for split in splits:
        assert (np.where(np.all(np.isclose(split[None,...], splits), axis=(1, 2)))[0]).size == 1

def test_get_max_shape():
    """
    Tests the get_max_shape() function of the DataProcessor class
    """
    # set seed
    seed = 11

    # get the instance
    data_processor = DataProcessor(np_random_seed=seed)

    # create test images
    test_imgs = [np.random.normal(size=(12,18)),
                 np.random.normal(size=(111,201)),
                 np.random.normal(size=(121,182)),
                 np.random.normal(size=(12,180))]

    # get the max shape
    divisor = 16
    height_max, width_max = data_processor.get_max_shape(imgs=test_imgs, divisor=divisor)

    # check that we get the expected results
    assert height_max == 128
    assert width_max == 208

def test_data_augmentation():
    """
    Tests the data_augmentation() function of the DataProcessor class
    """

    # set seed
    seed = 11

    # get the instance
    data_processor = DataProcessor(np_random_seed=seed)

    # create test data
    n = 10
    imgs = data_processor.scale_pixel_vals(np.random.normal(size=(n, 3, 4)))
    masks = np.random.normal(size=(n, 3, 4))
    weight_maps = np.random.normal(size=(n, 3, 4))
    splitting_cells = np.random.normal(size=n)

    # augment the data
    i_out, m_out, w_out, s_out = data_processor.data_augmentation(imgs=imgs, masks=masks, weight_maps=weight_maps,
                                                                  splitting_cells=splitting_cells)

    # check the shapes
    assert i_out.shape == (n * 9, 3, 4)
    assert m_out.shape == (n * 9, 3, 4)
    assert w_out.shape == (n * 9, 3, 4)
    assert s_out.shape == (n * 9, )

    # check the uniqueness of the imgs
    for i in i_out:
        assert (np.where(np.all(np.isclose(i[None, ...], i_out), axis=(1, 2)))[0]).size == 1

def test_run(monkeypatch):
    """
    Tests the data_run() function of the DataProcessor class
    """

    # set seed
    seed = 11

    # get the instance
    n_grid = 4
    num_split_c = 3
    num_split_r = 3
    patch_size = 4
    test_size = 0.2
    val_size = 0.1
    data_processor = DataProcessor(np_random_seed=seed, num_split_c=num_split_c, num_split_r=num_split_r,
                                   n_grid=n_grid, patch_size=patch_size, test_size=test_size, val_size=val_size,
                                   augment_patches=False)

    # This is a helper function
    def data_loader(name, *args, **kwargs):
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

        if "img" in name:
            return test_img
        if "mask" in name:
            return test_mask
        else:
            raise ValueError(f"Unkown name: {name}")

    # run the data_processor
    monkeypatch.setattr(io, 'imread', data_loader)
    data_out = data_processor.run(path_img="img", path_mask="mask")

    # check the length
    assert len(data_out) == 12
    # check a random dimension, the dimension can calculated from the test set and n_grid etc
    # we get the test size
    n_test = int(n_grid*n_grid*test_size) + 1
    n_val = int((n_grid*n_grid - n_test)*val_size) + 1
    n_train = n_grid*n_grid - n_test - n_val
    # multiy with the random patch gen
    n_train *= num_split_r*num_split_c
    assert data_out["X_train"].shape == (n_train, patch_size, patch_size, 1)

    # now with data augmentation
    data_processor = DataProcessor(np_random_seed=seed, num_split_c=num_split_c, num_split_r=num_split_r,
                                   n_grid=n_grid, patch_size=patch_size, test_size=test_size, val_size=val_size,
                                   augment_patches=True)
    data_out = data_processor.run(path_img="img", path_mask="mask")
    # check the length
    assert len(data_out) == 12
    # check a random dimension, the dimension can calculated from the test set and n_grid etc
    # 9 is added for data augmentation
    n_train *= 9
    assert data_out["X_train"].shape == (n_train, patch_size, patch_size, 1)

def test_run_mother_machine(monkeypatch):
    """
    Tests the data_run() function of the DataProcessor class
    """

    # set seed
    seed = 11

    # get the instance
    n_grid = 4
    num_split_c = 3
    num_split_r = 3
    patch_size = 4
    test_size = 0.2
    val_size = 0.1
    data_processor = DataProcessor(np_random_seed=seed, num_split_c=num_split_c, num_split_r=num_split_r,
                                   n_grid=n_grid, patch_size=patch_size, test_size=test_size, val_size=val_size,
                                   augment_patches=False)

    # This is a helper function
    def data_loader(name, *args, **kwargs):
        """
        A mock dataloader to test the run function
        :param name: The name of the dataset to load
        :param args: additional args (not used)
        :param kwargs: additional kwargs (not used)
        :return: The dataset
        """

        # fake data
        test_img = np.random.normal(size=(128, 16))
        test_mask = (test_img > 0.2).astype(int)

        if "img" in name:
            return test_img
        if "mask" in name:
            return test_mask
        else:
            raise ValueError(f"Unkown name: {name}")

    # override the imread
    monkeypatch.setattr(io, 'imread', data_loader)

    # monkey patch for list dir
    def fake_list(directory):
        """
        A fake function for the run_mother_machine test
        :param directory: A directory to list
        :return: Some files that can be loaded with the monkeypatch above
        """

        if "img" in directory:
            return ["img", "img", "img"]
        if "mask" in directory:
            return ["mask", "mask", "mask"]

    # override the listdir
    monkeypatch.setattr(os, "listdir", fake_list)

    X_train, y_train, X_val, y_val = data_processor.run_mother_machine(path_img="img", path_mask="mask")

    # check a random dimension, the dimension can calculated from the test set and n_grid etc
    # we get the test size (number images is the input to split)
    n_test = int(3*test_size) + 1
    n_train = 3 - n_test
    # multiy with the random patch gen (n splits is 150)
    n_train *= 150
    # height is a quarter of the input image, width stays the same
    assert X_train.shape == (n_train, 32, 16)

    # now with data aug
    data_processor = DataProcessor(np_random_seed=seed, num_split_c=num_split_c, num_split_r=num_split_r,
                                   n_grid=n_grid, patch_size=patch_size, test_size=test_size, val_size=val_size,
                                   augment_patches=True)
    X_train, y_train, X_val, y_val = data_processor.run_mother_machine(path_img="img", path_mask="mask")

    # times 9 for augment
    n_train *= 9

    # check shape
    assert X_train.shape == (n_train, 32, 16)
    