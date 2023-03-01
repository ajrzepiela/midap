import tempfile
from pathlib import Path

import pytest
import numpy as np
from skimage import io

"""
This file contains fixures that are shared between the tests in this directory
"""


@pytest.fixture()
def tmpdir():
    """
    Creates a temporary directory and returns the value
    :return: The path to the temporary directory as pathlib path
    """

    # create
    tmp_dir = tempfile.TemporaryDirectory()

    # yield
    yield Path(tmp_dir.name)

    # cleanup
    tmp_dir.cleanup()

@pytest.fixture()
def dir_setup(tmpdir):
    """
    Sets up a directory that can be used to test the DataProcessor
    :param tmpdir: A tmp dir to create the files
    :return: A tuple (tmpdir, paths) of pathlib Path objects, the first indicating the tmp directory
             the second is a list of paths that can be used for the DataProcessor init function
    """

    # fake data
    test_img = np.ones((5, 5))
    test_img = np.pad(test_img, [(5, 5), (5, 5)], mode='constant', constant_values=0.0)
    test_img = np.concatenate([test_img for i in range(20)], axis=1)
    test_img = np.concatenate([test_img for i in range(20)], axis=0)

    # the test mask is missing one cell
    test_mask = test_img.copy()
    test_mask[:10, :10] = 0

    # create the necessary files
    img = tmpdir.joinpath("img_raw.tif")
    io.imsave(img, test_img)
    seg = tmpdir.joinpath("img_seg.tif")
    io.imsave(seg, test_mask)
    paths = [img]

    return tmpdir, paths