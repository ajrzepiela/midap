import skimage.io as io
import numpy as np

from midap.tracking.deltav2_tracking import DeltaV2Tracking
from pytest import fixture
from pathlib import Path

# Fixtures
##########

@fixture()
def img1():
    """
    Creates a test image
    :return: A test image used as base image (no split, siingle cell)
    """
    # define the images
    img = np.zeros((512, 512))

    # create the cell
    img[25:100, 75:100] = 1

    return img

@fixture()
def img2(img1):
    """
    Creates a test image
    :return: A test image containing two cells
    """
    # define the images
    img = img1.copy()

    # now we create a split event
    img[55:65, 75:100] = 0

    return img

@fixture()
def tracking_instance(monkeypatch, img1, img2):
    """
    This fixture prepares the DeltaV2Tracking class including monkey patching for the image read
    :param monkeypatch: The monkeypatch fixture from pytest to override methods
    :param img1: A test image fixture used as base image to segment and track (single cell)
    :param img2: A test image fixture used as base image to segment and track (two cells)
    :return: A DeltaV1Tracking instance
    """

    def fake_load(path):
        """
        This is a monkeypatch for io.imread
        :param path: Path of the image to load
        :return: A loaded image
        """

        if "frame1" in path or "frame2" in path:
            return img1
        else:
            return img2

    # patch
    monkeypatch.setattr(io, "imread", fake_load)

    # prep the images
    imgs = ["img_frame1.png", "img_frame2.png", "img_frame3.png"]
    segs = ["seg_frame1.png", "seg_frame2.png", "seg_frame3.png"]

    # the model weights
    weight_path = Path(__file__).absolute().parent.parent.parent
    weight_path = weight_path.joinpath("model_weights", "model_weights_tracking", "unet_pads_track.hdf5")

    # sizes
    target_size = None
    input_size = None

    # get the instance
    deltav1 = DeltaV2Tracking(imgs=imgs, segs=segs, model_weights=weight_path, input_size=input_size,
                              target_size=target_size, connectivity=1)

    return deltav1

# Tests
#######

def test_run_model_crop(tracking_instance):
    """
    Tests the track_all_frames_crop routine with the DeltaV2Tracking class
    :param tracking_instance: A pytest fixture of an DeltaV2Tracking instance
    """

    inputs, results_all = tracking_instance.run_model_crop()

    # we should have 1 less result than number of input frames
    assert len(results_all) == 2

    # The first should not have a split
    first_res = results_all[0]
    assert first_res.shape == (512, 512, 2)
    assert first_res[..., 0].sum() != 0
    assert first_res[..., 1].sum() == 0

    # The second should have a split
    second_res = results_all[1]
    assert second_res.shape == (512, 512, 2)
    assert second_res[..., 0].sum() != 0
    assert second_res[..., 1].sum() != 0
