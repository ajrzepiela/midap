from configparser import ConfigParser
from typing import Optional

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

        # set the defaults
        self.set_defaults()

        # update
        overwrite = {k: v for k, v in zip(self.sections(), [general, cut_img, segmentation, tracking]) if v is not None}
        self.read_dict(overwrite)

    def set_defaults(self):
        """
        Sets all values of the Config to the default values
        """

        self.read_dict({"General": {"RunOption": "both",
                                    "Deconvolution": "no_deconf",
                                    "StartFrame": 0,
                                    "EndFrame": 10,
                                    "DataType": "Family_Machine",
                                    "FolderPath": "None",
                                    "FileType": "tif",
                                    "PosIdentifier": "pos",
                                    "Channels": 'None'
                                    },
                        "CutImg": {},
                        "Segmentation": {},
                        "Tracking": {}})

