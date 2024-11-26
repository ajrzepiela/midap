"""
This is a pytest configuration to share fixtures between files inside this directory
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture()
def setup_dir():
    """
    This fixture creates a temporary directory and creates all the necessary paths for the pipeline checks
    :return: The directory name
    """

    # create the directory
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name)

    # create all subdirs
    channel = "GFP"
    tmpdir_path.joinpath(channel, "raw_im").mkdir(parents=True)
    tmpdir_path.joinpath(channel, "cut_im").mkdir(parents=True)
    tmpdir_path.joinpath(channel, "cut_im_rawcounts").mkdir(parents=True)
    tmpdir_path.joinpath(channel, "seg_im").mkdir(parents=True)
    tmpdir_path.joinpath(channel, "seg_im_bin").mkdir(parents=True)
    tmpdir_path.joinpath(channel, "track_output").mkdir(parents=True)

    yield tmpdir.name, channel

    # cleanup
    tmpdir.cleanup()
