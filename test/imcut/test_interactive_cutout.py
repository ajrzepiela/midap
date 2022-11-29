import skimage.io as io
import numpy as np
import tempfile
import os

from midap.imcut.interactive_cutout import InteractiveCutout
from skimage.io import imread
from pytest import fixture
from os import listdir

# Fixtures
##########

@fixture()
def img1():
    """
    Creates a test image
    :return: A test image used as base image
    """
    # define the images
    img = np.array([[0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0]])

    # we pad the image with 0s on all sides to deal with the offset of the cutout
    pad_img = np.pad(img, pad_width=[[10, 10], [10, 10]], mode="constant", constant_values=0.0)

    return pad_img

@fixture()
def img2():
    """
    Creates a test image
    :return: A test image used as base image
    """
    # define the images
    img = np.array([[0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0]])

    # we pad the image with 0s on all sides to deal with the offset of the cutout
    pad_img = np.pad(img, pad_width=[[10, 10], [10, 10]], mode="constant", constant_values=0.0)

    return pad_img

@fixture()
def cutout_instance(monkeypatch, img1, img2):
    """
    This fixture prepares the InteractiveCutout class including monkey patching for the image read
    :param monkeypatch: The monkypatch fixture from pytest to override methods
    :param img1: A test image fixture used as base image
    :param img2: A test image fixture used as secondary images
    :return: A CutoutImage instance
    """

    # create the settings file
    with open("settings.sh", "w+") as f:
        pass

    # create a temp directory
    tmpdir = tempfile.TemporaryDirectory()

    # directories for the read and write
    img_dir = tmpdir.name
    cut_dir = os.path.join(tmpdir.name, "cut_im")
    os.makedirs(cut_dir)

    # thefine the paths
    channels = [os.path.join(img_dir, f"channel_{i}") for i in range(3)]

    def fake_list(directory):
        """
        Monkeypatch for os.listdir
        :param directory: path of the direcotry
        :return: A list of directories depending on the input
        """

        # we return different images depending on the channel
        if "channel_1" in directory:
            return ["img1.png", "img2.png", "img3.png", "img4.png"]
        else:
            return ["img1.png", "img2.png", "img3.png"]

    # patch
    monkeypatch.setattr(os, "listdir", fake_list)

    def fake_load(path):
        """
        This is a monkeypatch for io.imread
        :param path: Path of the image to load
        :return: A loaded image
        """

        if "img1" in path:
            return img1
        else:
            return img2

    # patch
    monkeypatch.setattr(io, "imread", fake_load)

    # get the instance
    cutout = InteractiveCutout(paths=channels)

    yield cutout

    # clean up
    tmpdir.cleanup()

    # remove the settings file again
    os.remove("settings.sh")

# Tests
#######

def test_align_two_images(cutout_instance, img1, img2):
    """
    Tests the align_two_images of the InteractiveCutout class
    :param cutout_instance: A pytest fixture returning an instance the class
    :param img1: A test image fixture used as base image
    :param img2: A test image fixture used as secondary images
    """

    # align
    alignment = cutout_instance.align_two_images(img1, img2)
    assert np.all(alignment == np.array([-1, -1]))

def test_align_all_images(cutout_instance):
    """
    Tests the align_all_images of the InteractiveCutout class
    :param cutout_instance: A pytest fixture returning an instance the class
    """

    cutout_instance.align_all_images()

    # check all shifts
    assert len(cutout_instance.shifts) == 2
    assert np.all([np.all(alignment == np.array([-1, -1])) for alignment in cutout_instance.shifts])
    # check offset
    assert cutout_instance.off

def test_save_corners(cutout_instance):
    """
    Tests the align_all_images of the InteractiveCutout class
    :param cutout_instance: A pytest fixture returning an instance the class
    """

    # set the corners
    cutout_instance.corners_cut = (1, 2, 3, 4)

    # save the corners
    cutout_instance.save_corners()

    # check the file
    with open("settings.sh", "r") as f:
        content = f.read()

    assert content == "CORNERS=(1 2 3 4)\n"

    # overwrite
    # set the corners
    cutout_instance.corners_cut = (11, 22, 33, 44)

    # save the corners
    cutout_instance.save_corners()

    # check the file
    with open("settings.sh", "r") as f:
        content = f.read()

    assert content == "CORNERS=(11 22 33 44)\n"

def test_run_align_cutout(monkeypatch, cutout_instance):
    """
    Tests the run_align_cutout of the InteractiveCutout class
    :param monkeypatch: The monkeypatch fixture from pytest to override methods
    :param cutout_instance: A pytest fixture returning an instance the class
    """

    def dummy_cutout(*args, **kwargs):
        """
        A monkeypatch to override the interactive cutout
        """

        # set the corners
        monkeypatch.setattr(cutout_instance, "corners_cut", (1, 5, 1, 5), raising=False)

    # monkey patch
    monkeypatch.setattr(cutout_instance, "cut_corners", dummy_cutout)

    # run the stack
    cutout_instance.run_align_cutout()

    # now we read in the images again
    dir_name = os.path.dirname(os.path.dirname(cutout_instance.channels[0][0]))
    dir_name = os.path.join(dir_name, "cut_im")
    imgs = []
    for fname in listdir(dir_name):
        imgs.append(imread(os.path.join(dir_name, fname)))

    # test if all are equal
    assert len(imgs) == 3
    assert np.allclose(np.array(imgs), imgs[0][None,...])
