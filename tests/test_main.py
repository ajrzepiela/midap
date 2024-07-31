import os
import tempfile
from pathlib import Path
from shutil import copyfile

import pytest

from midap.checkpoint import Checkpoint
from midap.config import Config
from midap.main import run_module


# Fixtures
##########

@pytest.fixture()
def prep_dir():
    """
    Creates a tmp dir and changes the working directory for a test and deletes it afterwards
    :return: The name of the directory as a string
    """

    # create
    tmpdir = tempfile.TemporaryDirectory()
    # get the current working dir
    current_dir = os.getcwd()
    # change to tmpdir
    os.chdir(tmpdir.name)

    # the name
    yield tmpdir.name

    # go back
    os.chdir(current_dir)
    # clean up
    tmpdir.cleanup()


@pytest.fixture()
def prep_settings(prep_dir):
    """
    Given a temporary directory, this fixture prepares it such that one can run the pipeline fully
    :param prep_dir: A fixture that sets up the tmp directory and makes it the work dir
    :return: The same path as prep_dir
    """

    # create the path for the data
    data_path = Path(prep_dir).joinpath("data")
    data_path.mkdir()

    # copy the test data
    src = Path(__file__).parent.absolute().joinpath("apps", "data", "example_stack.tiff")
    dst = data_path.joinpath("example_stack_pos1_PH.tiff")
    copyfile(src=src, dst=dst)

    # create a config
    config = Config(fname="settings.ini")
    config.set("General", "FolderPath", f"{data_path}")
    config.set("General", "FileType", "tiff")
    config.set("General", "IdentifierFound", "pos1")
    config.set_id_section("pos1")
    config.set("pos1", "StartFrame", "3")
    config.set("pos1", "EndFrame", "6")
    config.set("pos1", "Channels", "PH")
    config.set("pos1", "Corners", "7,102,68,155")
    path_model_weights = Path(__file__).parent.parent.joinpath("model_weights",
                                                               "model_weights_legacy",
                                                               "model_weights_C-crescentus-CB15_mKate2_v01.h5")
    config.set("pos1", "ModelWeights_PH", f"{path_model_weights}")
    config.to_file()

    return prep_dir


# Tests
#######

def test_run_module_create_config(prep_dir):
    """
    Tests the --create_config argument of the main routine of the package
    :param prep_dir: A fixture that sets up the tmp directory and makes it the work dir
    """

    # run the package
    run_module(["--create_config"])

    # check the existence
    assert Path("settings.ini").exists()


def test_run_module_restart(prep_dir):
    """
    Tests the --restart argument of the main routine of the package
    :param prep_dir: A fixture setting up the tmp directory and making it the work dir
    """

    # if we just run the module we should get a no checkpoint found error
    with pytest.raises(FileNotFoundError):
        run_module(["--restart"])

    # now we create a checkpoint
    checkpoint = Checkpoint("checkpoints.log")
    checkpoint.to_file()
    # we should still get an error because there is no config file
    with pytest.raises(FileNotFoundError):
        run_module(["--restart"])

    # now we also create a config file
    config = Config("settings.ini")
    config.set("General", "FolderPath", prep_dir)
    config.set("General", "IdentifierFound", "pos1")
    config.set_id_section("pos1")
    config.to_file()
    # create the dir of the pos1 identifier
    Path(prep_dir).joinpath("pos1").mkdir()
    # make the checkpoint such that all other things are skipped
    checkpoint.set_state(state="impossible state", identifier="impossible identifier")
    run_module(["--restart"])

    # now we create another directory
    another_dir = Path(prep_dir).joinpath("another_dir")
    another_dir.mkdir()
    # save the checkpoint there
    checkpoint.to_file(fname=another_dir)
    # we should get a value error because there are two checkpoints now
    with pytest.raises(ValueError):
        run_module(["--restart"])

    # if we save the config and set the path it should work
    config.to_file(another_dir)
    run_module(["--restart", f"{another_dir}"])


def test_run_module_full(prep_settings):
    """
    A test that runs the entire pipeline in headless mode and different settings
    :param prep_settings: A fixture that sets a prepared temporary directory as the current working dir
    """

    # to path for easier transformations
    prep_settings = Path(prep_settings)

    # run in headless mode
    run_module(["--headless"])
    # since we only have one channel and no phase seg, we expect a log of dirs to be empty
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "seg_im").iterdir())) == 0
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "seg_im_bin").iterdir())) == 0
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "track_output").iterdir())) == 0
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "cut_im").iterdir())) == 3
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "raw_im").iterdir())) == 3
    # checkpoint and config
    assert not prep_settings.joinpath("data", "pos1", "checkpoints.log").exists()
    assert prep_settings.joinpath("data", "pos1", "settings.ini").exists()

    # now we set phase segmentation and restart
    config = Config.from_file("settings.ini")
    config.set("pos1", "PhaseSegmentation", "True")
    config.to_file()
    run_module(["--headless"])

    # check everything again
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "seg_im").iterdir())) == 3
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "seg_im_bin").iterdir())) == 3
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "track_output").iterdir())) == 5
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "cut_im").iterdir())) == 3
    assert len(list(prep_settings.joinpath("data", "pos1", "PH", "raw_im").iterdir())) == 3
    # checkpoint and config
    assert not prep_settings.joinpath("data", "pos1", "checkpoints.log").exists()
    assert prep_settings.joinpath("data", "pos1", "settings.ini").exists()
