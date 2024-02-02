from pathlib import Path

import numpy as np
import skimage.io as io
from pytest import fixture

from midap.tracking.bayesian_tracking import BayesianCellTracking


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
    img[75:125, 90:100] = 1

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
    img[115:117, 90:100] = 0

    return img


@fixture()
def tracking_instance(monkeypatch, img1, img2):
    """
    This fixture prepares the BayesianCellTracking class including monkey patching for the image read
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
    imgs = ["img_frame1.png", "img_frame2.png", "img_frame3.png", "img_frame4.png"]
    segs = ["seg_frame1.png", "seg_frame2.png", "seg_frame3.png", "seg_frame4.png"]

    # the model weights (this is a dummy for v1 tracking but we set it corret anyway)
    weight_path = Path(__file__).absolute().parent.parent.parent
    weight_path = weight_path.joinpath("model_weights", "model_weights_tracking", "unet_moma_track_multisets.hdf5")

    # sizes
    target_size = None
    input_size = None

    # get the instance
    bayes = BayesianCellTracking(imgs=imgs, segs=segs, model_weights=weight_path, input_size=input_size,
                                 target_size=target_size)

    return bayes


# Tests
#######


def test_run_model_crop(tracking_instance):
    """
    Tests the track_all_frames_crop routine with the BayesianCellTracking class
    :param tracking_instance: A pytest fixture of an BayesianCellTracking instance
    """
    tracks = tracking_instance.run_model()
    df, label_stack = tracking_instance.generate_midap_output(tracks=tracks)

    # check the number of cells
    assert len(df) == 6

    # The first cell splits into cells with IDs 2 and 3, but bayes does not get it
    track_output_red = df[df["trackID"] == 1]
    assert track_output_red["trackID_d1"].isna().all()
    assert track_output_red["trackID_d2"].isna().all()

    # make sure the labelstack says the same
    assert np.unique(label_stack).size == 3