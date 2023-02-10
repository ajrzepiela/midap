from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pytest
from pytest import mark

from midap.apps.track_cells import main


# Fixtures
##########

@pytest.fixture()
@mark.usefixtures("setup_dir")
def prep_dirs(setup_dir):
    """
    A fixture that copies the segmented images into the temp directory for analysis
    :param setup_dir: The path to the temp directory containing the setup and the channel
    :return: The path to the channel directory
    """
    # unpack
    tmpdir_name, channel = setup_dir

    # copy all the files for the segmentation
    data_path = Path(__file__).parent.joinpath("data")
    src_files = data_path.joinpath("seg_im").glob("*[5-6]*.tif")
    channel_dir = Path(tmpdir_name).joinpath(channel)
    dst_dir = channel_dir.joinpath("seg_im")
    for f in src_files:
        copyfile(src=f, dst=dst_dir.joinpath(f.name))

    # copy all the files for the cut images
    src_files = data_path.joinpath("cut_im").glob("*[5-6]*.png")
    channel_dir = Path(tmpdir_name).joinpath(channel)
    dst_dir = channel_dir.joinpath("cut_im")
    for f in src_files:
        copyfile(src=f, dst=dst_dir.joinpath(f.name))

    return channel_dir


# Tests
#######

def test_main(prep_dirs):
    """
    Tests the main routine of the segment_analysis app
    :param prep_dirs: The prep dirs fixtures which creates everything necessary for the analyzation
    """

    # get the track output path
    track_output = prep_dirs.joinpath("track_output")

    # with wrong class
    with pytest.raises(ValueError):
        main(path=prep_dirs, tracking_class="This is not a valid class")

    # deltav1 tracking
    main(path=prep_dirs, tracking_class="DeltaV1Tracking")

    # check number of frames and number of cells
    delta_track_file = track_output.joinpath("track_output_delta.csv")
    res_df = pd.read_csv(delta_track_file)

    assert np.unique(res_df["frame"]).size == 2
    assert len(res_df) == 58

    # deltav2 tracking
    delta_track_file.unlink()
    main(path=prep_dirs, tracking_class="DeltaV2Tracking")

    # check number of frames and number of cells
    res_df = pd.read_csv(delta_track_file)
    assert np.unique(res_df["frame"]).size == 2
    assert len(res_df) == 58

    # bayes tracking
    main(path=prep_dirs, tracking_class="BayesianCellTracking")

    #Check bayes tracking results
    bayes_track_file = track_output.joinpath("track_output_bayesian.csv")
    res_df = pd.read_csv(bayes_track_file)
    assert np.unique(res_df["frame"]).size == 2
    assert len(res_df) == 55

