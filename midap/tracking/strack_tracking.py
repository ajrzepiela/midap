import os
from pathlib import Path
from typing import Union

import numpy as np

from .base_tracking import Tracking
from .strack.strack_script import run_strack
from .strack_lineages import STrackLineage


class STrack(Tracking):
    """
    A class for cell tracking using Bayesian tracking
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DeltaV2Tracking using the base class init
        :*args: Arguments used for the base class init
        :**kwargs: Keyword arguments used for the baseclass init
        """

        # base class init
        super().__init__(*args, **kwargs)

        # read the files
        raws = []
        segs = []
        for i in range(self.num_time_steps):
            r, _, s, _ = self.load_data(i, label=True)
            raws.append(r)
            segs.append(s)
        self.seg_imgs = np.array(segs)
        self.raw_imgs = np.array(raws)

    def track_all_frames(self, output_folder: Union[str, bytes, os.PathLike], max_dist=50.0, max_angle=30.0): #40
        """
        Tracks all frames and converts output to standard format.
        :param output_folder: Folder for the output
        :param max_dist: Maximum distance for linking (defaults to STrack default 50.0)
        :param max_angle: Maximum angle for linking (defaults to STrack default 30.0)
        """

        # create the strack directory
        output_folder = Path(output_folder)
        strack_dir = output_folder.joinpath("STrack")
        os.makedirs(strack_dir, exist_ok=True)
        run_strack(files_list=self.segs, output_dir=strack_dir, max_dist=max_dist, max_angle=max_angle)
        strack_lineages = STrackLineage(output_folder, imgs=self.raw_imgs, segs=self.seg_imgs)
        data_file, csv_file = strack_lineages.generate_midap_output()

        return data_file, csv_file
