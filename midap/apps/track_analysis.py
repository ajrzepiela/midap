import numpy as np
import os
from pathlib import Path
from typing import Union, List

from midap.tracking.tracking_analysis import FluoChangeAnalysis


def main(path: Union[str, os.PathLike], channels: List[str], tracking_class: str):
    """
    Loads tracking output and adds intensities per cell and channel to dataframe.
    :param path: Path to output folder.
    :param channels: List with channels.
    :param tracking_class: Name of used tracking class.
    """
    fca = FluoChangeAnalysis(path, channels, tracking_class)
    fca.gen_pathnames()
    fca.load_images()
    fca.gen_column_names()

    df_fluo_change = fca.create_output_df(fca.path_ref_csv)
    df_fluo_change = fca.add_fluo_intensity(df_fluo_change)

    fca.save_fluo_change(df_fluo_change)


if __name__ == "__main__":
    path = Path(
        "/Users/franziskaoschmann/Documents/midap-datasets/Simon-GlucoseAcetate/GlucoseAcetateExampleData/Data/pos1"
    )
    channels = ["ph", "gfp", "mCherry"]
    tracking_class = "STrack"
    main(path, channels, tracking_class=tracking_class)
