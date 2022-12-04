import btrack
from btrack import datasets

import numpy as np

#from .model_tracking_bayesian import BayesianCellTracking
from .base_tracking import Tracking

class BayesianCellTracking(Tracking):
    """
    A class for cell tracking using the U-Net Delta V2 model

    ...

    Attributes
    ----------
    imgs: list of str
        list of path strings
    segs : list of strs
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

    def __init__(self, *args, **kwargs): #input_type, output_type, 
        """
        Initializes the DeltaV2Tracking using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the basecalss init
        """

        # base class init
        super().__init__(*args, **kwargs)


    def set_params(self):

        self.features = ["area", "major_axis_length", "minor_axis_length", "orientation"]

        self.objects = btrack.utils.segmentation_to_objects(
            self.seg_imgs, 
            properties=tuple(self.features), 
        )

        self.config_file = datasets.cell_config()


    def run_tracking(self):
        """Loads model for inference/tracking.

        Parameters
        ----------
        constant_input: array, optional
            Array containing the constant input (whole raw image and segmentation image) per time frame.
        """
        # for cur_frame in (pbar := tqdm(range(1, self.num_time_steps), postfix={"RAM": f"{ram_usg:.1f} GB"})):
        #     _, _, seg, _ = self.load_data(
        #         cur_frame)
        self.seg_imgs = np.array([self.load_data(cur_frame)[2] for cur_frame in range(1, self.num_time_steps)])

        self.set_params()
        # initialise a tracker session using a context manager
        with btrack.BayesianTracker() as self.tracker:

            # configure the tracker using a config file
            self.tracker.configure_from_file(self.config_file)
            self.tracker.verbose = True
            self.tracker.max_search_radius = 50
            self.tracker.features = self.features

            # append the objects to be tracked
            self.tracker.append(self.objects)

            # set the tracking volume
            self.tracker.volume=((0, 1600), (0, 1200))

            # track them (in interactive mode)
            self.tracker.track_interactive(step_size=100)

            # generate hypotheses and run the global optimizer
            self.tracker.optimize()
