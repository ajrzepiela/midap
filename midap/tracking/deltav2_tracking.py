from .model_tracking import unet_track
from .base_tracking import Tracking

class DeltaV2Tracking(Tracking):
    """
    A class for cell tracking using the U-Net Delta V2 model

    ...

    Attributes
    ----------
    imgs: list of str
        list of path strings
    segs : list of str
        list of path strings
    model_weights : str 
        path to model weights
    input_size : tuple
        input size of tracking network
    target_size: tuple
        target size of tracking network
    crop_size: tuple
        size of cropped input image

    Methods
    -------
    load_model(self)
        Loads model for inference/tracking.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DeltaV2Tracking using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """

        # base class init
        super().__init__(*args, **kwargs)

    def load_model(self):
        """Loads model for inference/tracking.

        Parameters
        ----------
        constant_input: array, optional
            Array containing the constant input (whole raw image and segmentation image) per time frame.
        """

        self.model = unet_track(self.model_weights, self.input_size)
