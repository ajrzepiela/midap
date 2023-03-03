import os

from configparser import ConfigParser
from pathlib import Path
from typing import Union, Optional
from copy import deepcopy
from .config import Config
from .utils import get_logger

# get the logger we readout the variable or set it to max output
if "__VERBOSE" in os.environ:
    loglevel = int(os.environ["__VERBOSE"])
else:
    loglevel = 7
logger = get_logger(__file__, loglevel)

class Checkpoint(ConfigParser):
    """
    This class implements the checkpoint files of the MIDAP pipeline as simple config files
    """

    def __init__(self, fname: Union[str,Path,None]):
        """
        Inits the checkpoint with a given file name. A default restart point is created.
        :param fname: The name of the checkpoint file.
        """

        # init the parser
        super().__init__()

        # make all keys case sensitive
        self.optionxform = str

        # save the file_name
        self.fname = fname

        # set the defaults
        self.set_defaults()

    def set_defaults(self):
        """
        Sets the defaults of the checkpoint (a dummy checkpoint)
        """

        self.read_dict({"Checkpoint": {"Function": "None"},
                        "Settings": {"Identifier": "None"}})

    def to_file(self, fname: Union[str,Path,None]=None, overwrite=True):
        """
        Write the config into a file
        :param fname: Name of the file to write, defaults to fname attribute. If a directory is specified, the file
                      will be saved in that directory with the same name, if a full path is specified, the full path
                      is used to save the file.
        :param overwrite: Overwrite existing file, defaults to True
        :raises: FileExistsError if overwrite is False and file exists
        """

        # check if we have an argument for the file name
        if fname is not None:
            fname = Path(fname)
            # if we have a dir we add the fname attribute
            if fname.is_dir():
                fname = fname.joinpath(self.fname)
        else:
            fname = Path(self.fname)

        # check
        if not overwrite and fname.exists():
            raise FileExistsError(f"File already exists, set overwrite to True to overwrite: {fname}")

        # now we can open a w+ without worrying
        with open(fname, "w+") as f:
            self.write(f)

    def get_state(self, identifier=False):
        """
        Shortcut to get("Checkpoint", "Function")
        :param identifier: If True, "Settings" "Identifier" is also returned
        :return: The current state of the checkpoint
        """

        if identifier:
            return self.get("Checkpoint", "Function"), self.get("Settings", "Identifier")
        else:
            return self.get("Checkpoint", "Function")

    def set_state(self, state: str, identifier="None", flush=True, **kwargs):
        """
        Sets the state of the checkpoint
        :param state: The state of the checkpoint
        :param identifier: The identifier to set
        :param flush: Save the checkpoint to file after update
        :param kwargs: Additional keyword arguments forwarded to the Settings section
        """

        # define the new state
        new_state = {"Checkpoint": {"Function": state},
                     "Settings": {"Identifier": identifier}}

        # update the settings if necessary
        new_state["Settings"].update(kwargs)

        # update the checkpoint state
        self.read_dict(new_state)

        # write to file
        if flush:
            self.to_file()

    @classmethod
    def from_file(cls, fname: Union[str,bytes,os.PathLike]):
        """
        Creates a Checkpoint instance from a file
        :param fname: The name of the file to read
        :return: An instance of the class
        """

        # get the path
        fname = Path(fname)

        # create a class instance
        if fname.is_file():
            checkpoint = Checkpoint(fname=fname.name)
        else:
            raise FileNotFoundError(f"File {fname} does not exist!")

        # read the file
        with open(fname, "r") as f:
            checkpoint.read_file(f)

        return checkpoint


class AlreadyDoneError(Exception):
    """
    A simple error raised by the CheckpointChecker if we do not need to rerun a block
    """

    def __init__(self, message="Already done this..."):
        super().__init__(message)


class CheckpointChecker(object):
    """
    This is a helper class that checks if we need to run something or we skip ip
    """
    def __init__(self, restart: bool, checkpoint: Checkpoint, state: str, identifier: str):
        """
        Create a CheckpointChecker to compare the current state of the checkpoint to the fail state and see if we need
        to run something
        :param restart: The restart flag of the pipeline, if False, then the code in the with block will always
                        be executed
        :param checkpoint: The checkpoint instance to use for the manager
        :param state: The new state of the checkpoint in case of failure
        :param identifier: The new identifier of the checkpoint in case of failure
        """

        # set the attributes
        self.restart = restart
        self.checkpoint = checkpoint
        self.state = state
        self.identifier = identifier

    def check(self):
        """
        Checks if we need to run something, raises a AlreadyDoneError if not
        :raises: AlreadyDoneError if the checkpoint state indicates that we do not need to run something
        """

        # if we are not in restart mode, we run everything, otherwise we rerun if new_state == old_state
        if self.restart:
            # we get the current state
            current_state = self.checkpoint.get_state(identifier=True)
            # if the Function value is None, we run everything
            if current_state == ("None", "None"):
                return None

            if current_state != (self.state, self.identifier):
                raise AlreadyDoneError(f"Already done this! State: {self.state}, Identifier: {self.identifier}")


class CheckpointManager(object):
    """
    This is a context manager
    """

    def __init__(self, restart: bool, checkpoint: Checkpoint, config: Config, state: str,
                 identifier: str, copy_path: Union[str,Path,None]=None):
        """
        Create a checkpoint manager to run things in a with statement for easy checkpointing
        :param restart: The restart flag of the pipeline, if False, then the code in the with block will always
                        be executed
        :param checkpoint: The checkpoint instance to use for the manager
        :param config: The Config instance to use
        :param state: The new state of the checkpoint in case of failure
        :param identifier: The new identifier of the checkpoint in case of failure
        :param copy_path: A path to copy the checkpoint and config to in case of failure
        """

        # set the attributes
        self.restart = restart
        self.checkpoint_original = deepcopy(checkpoint)
        self.checkpoint = checkpoint
        self.config = config
        self.state = state
        self.identifier = identifier
        self.copy_path = copy_path

    def __enter__(self):
        """
        Enter the CheckpointManager context, return a CheckpointChecker to check if we actually need to run something
        :return: A CheckpointChecker object
        """

        # update the checkpoint
        self.checkpoint.set_state(state=self.state, identifier=self.identifier)
        self.save_files()

        # the checkpoint checker needs to get the original checkpoint not the updated one
        return CheckpointChecker(restart=self.restart, checkpoint=self.checkpoint_original, state=self.state,
                                 identifier=self.identifier)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The teardown mechanism of the context manager
        :param exc_type: The Exception type, can be None
        :param exc_val: The Exception value, can be None
        :param exc_tb: The trace back
        :return: If there was an exception the method returns True if the exception was handled gracefully, otherwise
                 we do the teardown and the exception is forwarded
        """

        # if we already did it, there is nothing to do
        if isinstance(exc_val, AlreadyDoneError):
            logger.info(f"Skipping {self.state} for {self.identifier}...")
            # we save the original checkpoint
            original_state, original_identifier = self.checkpoint_original.get_state(identifier=True)
            self.checkpoint.set_state(state=original_state, identifier=original_identifier, flush=True)
            self.save_files()
            return True

        # if there is no Error and we successfully finished the job we reset the checkpoint
        if exc_val is None:
            self.checkpoint.set_state(state="None", identifier="None", flush=True)
            self.unlink_checkpoint(missing_ok=True)
            return True

        # we update the checkpoint if we fail otherwise
        if exc_type is not None:
            logger.info(f"Error while running {self.state} for {self.identifier}")
            return False

    def save_files(self):
        """
        Saves the checkpoint and settings, also makes a copy if copy_path is set
        """

        # save checkpoint and config
        self.checkpoint.to_file()
        self.config.to_file()

        # save a copy if we have a path
        if self.copy_path is not None:
            logger.info(f"Saving a copy of the checkpoint and settings to: {self.copy_path}")
            self.checkpoint.to_file(self.copy_path)
            self.config.to_file(self.copy_path)

    def unlink_checkpoint(self, missing_ok=True):
        """
        Removes the checkpoints from disk
        :param missing_ok: Argument forwarded tp Path.unlink
        """

        # the standard path
        original_path = Path(self.checkpoint.fname)
        original_path.unlink(missing_ok=missing_ok)

        # the copy one
        if self.copy_path is not None:
            Path(self.copy_path).joinpath(original_path.name).unlink(missing_ok=missing_ok)
