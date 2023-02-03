import os
import tempfile

import pytest

from midap.config import Config
from midap.checkpoint import Checkpoint, AlreadyDoneError, CheckpointChecker, CheckpointManager
from pathlib import Path

# Fixtures
##########

@pytest.fixture()
def tmp_dir():
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


# Tests
#######

def test_Checkpoint(tmp_dir):
    """
    Tests the functionality of the Checkpoint class
    """

    # get the class
    checkpoint = Checkpoint(fname="checkpoint.log")

    # check the settings
    assert checkpoint.get_state(identifier=True) == ("None", "None")

    # set a new state
    checkpoint.set_state(state="state", identifier="identifier", flush=False, foo="bar")
    assert checkpoint.get_state(identifier=True) == ("state", "identifier")
    assert checkpoint.get("Settings", "foo") == "bar"

    # write to file
    checkpoint.to_file(tmp_dir, overwrite=True)
    with pytest.raises(FileExistsError):
        checkpoint.to_file(tmp_dir, overwrite=False)

    # from file
    with pytest.raises(FileNotFoundError):
        _ = Checkpoint.from_file(os.path.join(tmp_dir, "checkpoint.gol"))
    new_checkpoint = Checkpoint.from_file(os.path.join(tmp_dir, "checkpoint.log"))

    # name and same as old
    assert new_checkpoint.fname == "checkpoint.log"
    assert new_checkpoint == checkpoint


def test_CheckpointChecker():
    """
    Test the CheckpointChecker class
    """

    # a checkpoint for the checker
    checkpoint = Checkpoint("checkpoint.log")

    # the state we are currently running
    state = "state"
    identifier = "identifier"

    # If we are not in restart mode, we run everything always
    checker = CheckpointChecker(restart=False, checkpoint=checkpoint, state=state, identifier=identifier)
    assert checker.check() is None

    # if we are in restart mode and the state is None we run the thing as well
    checker = CheckpointChecker(restart=True, checkpoint=checkpoint, state=state, identifier=identifier)
    assert checker.check() is None

    # Restart mode and non-matching states that are not None should throw an error
    checkpoint.set_state(state="not state", identifier="not identifier", flush=False)
    checker = CheckpointChecker(restart=True, checkpoint=checkpoint, state=state, identifier=identifier)
    with pytest.raises(AlreadyDoneError):
        checker.check()


def test_CheckpointManager(tmp_dir):
    """
    Tests the CheckpointManager class
    :param tmp_dir: A temporary directory to test the functionality
    """

    # prep
    checkpoint = Checkpoint("checkpoint.log")
    config = Config("settings.ini")
    state = "state"
    identifier = "identifier"

    # if we enter without restart, the checkpoint should reset to None in any case
    checkpoint.set_state(state=state, identifier=identifier)
    with CheckpointManager(restart=False, checkpoint=checkpoint, config=config, state=state,
                           identifier=identifier) as checker:
        checker.check()
    assert checkpoint.get_state(identifier=True) == ("None", "None")

    # if restart is set and the states match it should also reset
    checkpoint.set_state(state=state, identifier=identifier)
    with CheckpointManager(restart=True, checkpoint=checkpoint, config=config, state=state,
                           identifier=identifier) as checker:
        checker.check()
    assert checkpoint.get_state(identifier=True) == ("None", "None")

    # if there is an exception in the execution the checkpoint should stay
    with pytest.raises(ValueError):
        checkpoint.set_state(state=state, identifier=identifier)
        with CheckpointManager(restart=True, checkpoint=checkpoint, config=config, state=state,
                               identifier=identifier) as checker:
            checker.check()
            raise ValueError("Ooops")
    assert checkpoint.get_state(identifier=True) == (state, identifier)

    # if restart is set and the states do not match it should stay
    checkpoint.set_state(state="not state", identifier="not identifier")
    with CheckpointManager(restart=True, checkpoint=checkpoint, config=config, state=state,
                           identifier=identifier) as checker:
        checker.check()
        # This code should never execute, because we already did this
        assert False
    assert checkpoint.get_state(identifier=True) == ("not state", "not identifier")

    # Finally we check for the file existence after the run
    current_path = Path(tmp_dir)
    copy_path = current_path.joinpath("copy_path")
    copy_path.mkdir()
    checkpoint.set_state(state="not state", identifier="not identifier")
    with CheckpointManager(restart=True, checkpoint=checkpoint, config=config, state=state,
                           identifier=identifier, copy_path=copy_path) as checker:
        checker.check()
    assert current_path.joinpath("checkpoint.log").is_file()
    assert current_path.joinpath("settings.ini").is_file()
    assert copy_path.joinpath("checkpoint.log").is_file()
    assert copy_path.joinpath("settings.ini").is_file()
