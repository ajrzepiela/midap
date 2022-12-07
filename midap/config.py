import git

from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from typing import Optional

# Get all subclasses to check validity of config
################################################

# get all subclasses from the imcut
from midap.imcut import *
from midap.imcut import base_cutout

imcut_subclasses = [subclass.__name__ for subclass in base_cutout.CutoutImage.__subclasses__()]

# get all subclasses from the segmentations
from midap.segmentation import *
from midap.segmentation import base_segmentator

segmentation_subclasses = [subclass.__name__ for subclass in base_segmentator.SegmentationPredictor.__subclasses__()]

# get all subclasses from the tracking
from midap.tracking import *
from midap.tracking import base_tracking

tracking_subclasses = [subclass.__name__ for subclass in base_tracking.Tracking.__subclasses__()]

class Config(ConfigParser):
    """
    A subclass of the ConfigParser defining all values of the MIDAP pipeline.
    """

    def __init__(self, general: Optional[dict]=None, cut_img: Optional[dict]=None, segmentation: Optional[dict]=None,
                 tracking: Optional[dict]=None):
        """
        Initializes the Config of the pipeline, the default values of the sections are updated with the entries
        provided in the dictionary
        :param general: A dictionary used for the entries of the General section of the config
        :param cut_img: A dictionary used for the entries of the CutImg section of the config
        :param segmentation: A dictionary used for the entries of the Segmentation section of the config
        :param tracking: A dictionary used for the entries of the Tracking section of the config
        """

        # init the parser
        super().__init__()

        # make all keys case sensitive
        self.optionxform = str

        # set the defaults
        self.set_defaults()

        # update
        overwrite = {k: v for k, v in zip(self.sections(), [general, cut_img, segmentation, tracking]) if v is not None}
        self.read_dict(overwrite)

    def set_defaults(self):
        """
        Sets all values of the Config to the default values
        """

        # get the SHA of the git repo
        try:
            repo = git.Repo(path=Path(__file__).parent, search_parent_directories=True)
            sha = repo.head.object.hexsha
        except git.InvalidGitRepositoryError:
            sha = 'None'

        # set defaults
        self.read_dict({"General": {"Timestamp": datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),
                                    "Git hash": sha,
                                    "RunOption": "both",
                                    "Deconvolution": "no_deconv",
                                    "StartFrame": 0,
                                    "EndFrame": 10,
                                    "DataType": "Family_Machine",
                                    "FolderPath": "None",
                                    "FileType": "tif",
                                    "PosIdentifier": "pos",
                                    "PhaseSegmentation": False,
                                    "Channels": 'None'
                                    },
                        "CutImg": {"Class": "InteractiveCutout"},
                        "Segmentation": {"Class": "UNetSegmentation"},
                        "Tracking": {"Class": "DeltaV2Tracking"}})

    def validate_sections(self):
        """
        Validates the content of all sections in choice and type
        :raises: ValueError if invalid value is found or other Errors accordingly
        """

        # run option choices
        allowed_run_options = ['both', 'segmentation', 'tracking']
        if self.get("General", "RunOption").lower() not in allowed_run_options:
            raise ValueError(f"'RunOption' not in {allowed_run_options}")

        # deconvolution choices
        allowed_deconv = ["deconv_family_machine", "no_deconv"]
        if self.get("General", "Deconvolution").lower() not in allowed_deconv:
            raise ValueError(f"'Deconvolution' not in {allowed_deconv}")

        # check the DataType
        allowed_datatype = ["Family_Machine"]
        if self.get("General", "DataType") not in allowed_datatype:
            raise ValueError(f"'DataType' not in {allowed_datatype}")

        # check the ints
        if (start_frame := self.getint("General", "StartFrame")) < 0:
            raise ValueError(f"'StartFrame' has to be a positive integer, is: {start_frame}")
        if (end_frame := self.getint("General", "EndFrame")) < 0 or end_frame <= start_frame:
            raise ValueError(f"'EndFrame' has to be a positive integer and larger than 'StartFrame', is: {start_frame}")

        # check the booleans
        _ = self.getboolean("General", "PhaseSegmentation")

        # check the paths
        if not (folder_path := Path(self.get("General", "FolderPath"))).exists():
            raise FileNotFoundError(f"'FolderPath' not an existing directory: {folder_path}")

        # check all the classes
        if self.get("CutImg", "Class") not in imcut_subclasses:
            raise ValueError(f"'Class' of 'CutImg' not in {imcut_subclasses}")
        if self.get("Segmentation", "Class") not in segmentation_subclasses:
            raise ValueError(f"'Class' of 'Segmentation' not in {segmentation_subclasses}")
        if self.get("Tracking", "Class") not in tracking_subclasses:
            raise ValueError(f"'Class' of 'Tracking' not in {tracking_subclasses}")

    def to_file(self, fname, overwrite=True):
        """
        Write the config into a file
        :param fname: Name of the file to write
        :param overwrite: Overwrite existing file, defaults to True
        :raises: FileExistsError if overwrite is False and file exists
        """

        # check
        if not overwrite and Path(fname).exists():
            FileExistsError(f"File already exists, set overwrite to True to overwrite: {fname}")

        # now we can open a w+ without worrying
        with open(fname, "w+") as f:
            self.write(f)

    @classmethod
    def from_file(cls, fname):
        """
        Initiates a new instance of the class and overwrites the defaults with contents from a file. The contents read
        from the file will be checked for validity.
        :param fname: The name of the file to read
        :return: An instance of the class
        """

        # create a class instance
        config = Config()

        # read the file
        with open(fname, "r") as f:
            config.read_file(f)

        # check validity
        config.validate_sections()

        # if no error was thrown we return the instance
        return config
