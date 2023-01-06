from ..networks.deltav2 import unet_track
from .base_tracking import DeltaTypeTracking

class DeltaV2Tracking(DeltaTypeTracking):
    """
    A class for cell tracking using the U-Net Delta V2 model
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DeltaV2Tracking using the base class init
        :param args: Arguments used for the base class init
        :param kwargs: Keyword arguments used for the baseclass init
        """

        # base class init
        super().__init__(*args, **kwargs)

    def load_model(self):
        """
        Loads model for inference/tracking.
        """

        self.model = unet_track(self.model_weights, self.input_size)
