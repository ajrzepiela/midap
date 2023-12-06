import numpy as np
import pytest
from pytest import mark
from shutil import copyfile
from pathlib import Path

from midap.apps.cut_chamber import main
from skimage import io


# Fixtures
##########

@pytest.fixture()
@mark.usefixtures("setup_dir")
def prep_dirs(setup_dir):
    """
    A fixture that copies the raw images into the temp directory for cut outs
    :param setup_dir: The path to the temp directory containing the setup and the channel
    :return: The path to the directory containing the directories for the raw and cut images and the directory
             containing the solution
    """
    # unpack
    tmpdir_name, channel = setup_dir

    # copy all the files
    data_path = Path(__file__).parent.joinpath("data")
    src_files = data_path.joinpath("raw_im").glob("*.png")
    channel_dir = Path(tmpdir_name).joinpath(channel)
    dst_dir = channel_dir.joinpath("raw_im")
    for f in src_files:
        copyfile(src=f, dst=dst_dir.joinpath(f.name))

    return channel_dir, data_path.joinpath("cut_im")


# Tests
#######

def test_main(prep_dirs):
    """
    Tests the main routine of the cut_chamber app
    :param prep_dirs: Directory of the prepared channel containing the raw_im and cut_im directories
    """

    # unpack
    channel_dir, sol_path = prep_dirs

    # pre args
    channel = channel_dir.joinpath("raw_im")
    cutout_class = "InteractiveCutout"
    corners = (7, 102, 68, 155)

    # test error
    with pytest.raises(ValueError):
        main(channel=channel, cutout_class="This is not a valid class", corners=corners)
    # run the main
    main(channel=channel, cutout_class=cutout_class, corners=corners)

    # compare with the solution
    cut_imgs = sorted(channel_dir.joinpath("cut_im").glob("*.png"))
    sol_imgs = sorted(sol_path.glob("*.png"))

    # check same length
    assert len(cut_imgs) == len(sol_imgs)

    # check pixel
    for sol_f, cut_f in zip(sol_imgs, cut_imgs):
        sol = io.imread(sol_f)
        cut = io.imread(cut_f)
        assert np.allclose(sol, cut)
