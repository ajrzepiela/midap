import skimage.io as io
import pandas as pd
import numpy as np

from midap.tracking.bayesian_tracking import BayesianCellTracking
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
    img[25:200, 75:100] = 1

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
    img[100:105, 75:100] = 0

    return img


@fixture()
def img3(img2):
    """
    Creates a test image
    :return: A test image containing two cells
    """
    img = img2.copy()

    return img


@fixture()
def tracking_instance(monkeypatch, img1, img2, img3):
    """
    This fixture prepares the DeltaV1Tracking class including monkey patching for the image read
    :param monkeypatch: The monkeypatch fixture from pytest to override methods
    :param img1: A test image fixture used as base image to segment and track (single cell)
    :param img2: A test image fixture used as base image to segment and track (two cells)
    :param img3: A test image fixture used as base image to segment and track (two cells)
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
        elif "frame3" in path:
            return img2
        else:
            return img3

    # patch
    monkeypatch.setattr(io, "imread", fake_load)

    # prep the images
    imgs = ["img_frame1.png", "img_frame2.png", "img_frame3.png", "img_frame4.png"]
    segs = ["seg_frame1.png", "seg_frame2.png", "seg_frame3.png", "seg_frame4.png"]

    # the model weights (this is a dummy for v1 tracking but we set it corret anyway)
    weight_path = Path(__file__).absolute().parent.parent.parent
    weight_path = weight_path.joinpath(
        "model_weights", "model_weights_tracking", "unet_moma_track_multisets.hdf5"
    )

    # sizes
    crop_size = (128, 128)
    target_size = (512, 512)
    input_size = crop_size + (4,)

    # get the instance
    bayes = BayesianCellTracking(
        imgs=imgs,
        segs=segs,
        model_weights=weight_path,
        input_size=input_size,
        target_size=target_size,
        crop_size=crop_size,
    )

    return bayes


# Tests
#######


def test_run_model_crop(tracking_instance):
    """
    Tests the track_all_frames_crop routine with the DeltaV1Tracking class
    :param tracking_instance: A pytest fixture of an DeltaV1Tracking instance
    """

    tracking_instance.run_model()
    tracking_instance.convert_data()
    tracking_instance.generate_label_stack()
    tracking_instance.correct_label_stack()

    track_output_correct = tracking_instance.track_output_correct

    # The first cell splits into cells with IDs 2 ansd 3
    track_output_red = track_output_correct[track_output_correct["trackID"] == 1]
    assert track_output_red["trackID_d1"][0] == 3
    assert track_output_red["trackID_d2"][0] == 2

    # The mother ID should be the same for both daughter cells
    track_output_red_d1 = track_output_correct[track_output_correct["trackID_d1"] == 3]
    track_output_red_d2 = track_output_correct[track_output_correct["trackID_d2"] == 2]
    assert track_output_red_d1["trackID_mother"].values[0] == 1
    assert track_output_red_d2["trackID_mother"].values[0] == 1
    assert (
        track_output_red_d1["trackID_mother"].values[0]
        == track_output_red_d2["trackID_mother"].values[0]
    )
