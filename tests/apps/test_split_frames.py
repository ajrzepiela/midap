from pathlib import Path

import numpy as np
import skimage.io as io
from pytest import mark

from midap.apps.split_frames import main


@mark.usefixtures("setup_dir")
def test_main(setup_dir):
    """
    Tests the main routine of the split_frames app
    :param setup_dir: The path to the temp directory containing the setup and the channel
    """

    # unpack
    tmpdir_name, channel = setup_dir

    # arg setup
    path = Path(__file__).parent.joinpath("data", "example_stack.tiff")
    save_dir = Path(tmpdir_name).joinpath(channel, "raw_im")
    frames = [2, 5, 6]

    # split frames
    main(path=path, save_dir=save_dir, frames=frames, deconv="no_deconv")

    # check
    original_files = sorted(path.parent.joinpath("raw_im").glob("*.png"))
    new_files = sorted(save_dir.glob("*.png"))
    for original_fname, new_fname in zip(original_files, new_files):
        true_img = io.imread(original_fname)
        new_img = io.imread(new_fname)

        assert np.allclose(true_img, new_img)
