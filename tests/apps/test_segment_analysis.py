from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pytest
from pytest import mark

from midap.apps.segment_analysis import main


# Fixtures
##########


@pytest.fixture()
@mark.usefixtures("setup_dir")
def prep_dirs(setup_dir):
    """
    A fixture that copies the segmented images into the temp directory for analysis
    :param setup_dir: The path to the temp directory containing the setup and the channel
    :return: The path to the directory containing the segmented images and the path where the soltion should be
             written
    """
    # unpack
    tmpdir_name, channel = setup_dir

    # copy all the files
    data_path = Path(__file__).parent.joinpath("data")
    src_files = data_path.joinpath("seg_im").glob("*.tif")
    channel_dir = Path(tmpdir_name).joinpath(channel)
    dst_dir = channel_dir.joinpath("seg_im")
    for f in src_files:
        copyfile(src=f, dst=dst_dir.joinpath(f.name))

    return dst_dir, Path(tmpdir_name)


# Tests
#######


def test_main(prep_dirs):
    """
    Tests the main routine of the segment_analysis app
    :param prep_dirs: The prep dirs fixtures which creates everything necessary for the analyzation
    """

    # unpack args
    path_seg, path_result = prep_dirs

    # run the main
    main(path_seg=path_seg, path_result=path_result)

    # compare
    res_df = pd.read_csv(path_result.joinpath("cell_number.csv"))
    assert np.allclose(res_df["all cells"].values, [28, 30, 28])
