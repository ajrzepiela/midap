import tempfile
from pathlib import Path

import pytest

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

    # create the necessary files
    img = tmpdir.joinpath("img_raw.tif")
    img.touch()
    seg = tmpdir.joinpath("img_seg.tif")
    seg.touch()
    paths = [img]

    return tmpdir, paths