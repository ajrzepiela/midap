from configparser import ConfigParser
from pathlib import Path
from typing import Optional


class Checkpoint(ConfigParser):
    """
    This class implements the checkpoint files of the MIDAP pipeline as simple config files
    """

    def __init__(self, fname):
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
                        "Settings": {}})

    def to_file(self, fname: Optional[str]=None, overwrite=True):
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
            FileExistsError(f"File already exists, set overwrite to True to overwrite: {fname}")

        # now we can open a w+ without worrying
        with open(fname, "w+") as f:
            self.write(f)

    @classmethod
    def from_file(cls, fname: str):
        """
        Creates a Checkpoint instance from a file
        :param fname: The name of the file to read
        :return: An instance of the class
        """

        # create a class instance
        checkpoint = Checkpoint(fname=fname)

        # read the file
        with open(fname, "r") as f:
            checkpoint.read_file(f)

        return checkpoint
