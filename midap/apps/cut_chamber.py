import argparse
import os

# import to get all subclasses
from midap.imcut import *
from midap.imcut import base_cutout

from typing import Optional, Union, Iterable


def main(channel: Union[str, Iterable[str], bytes, Iterable[bytes], os.PathLike, Iterable[os.PathLike]],
         cutout_class: str, corners: Optional[tuple]=None):
    """
    Performs the image cutout and alignment on all images in the paths
    :param channel: A single directory or a list of directories with the images to cut and align
    :param cutout_class: Name of the class used to perform the chamber cutout. Must be defined in a file of
                         midap.imcut and a subclass of midap.imcut.base_cutout.CutoutImage
    """
    # get the right subclass
    class_instance = None
    for subclass in base_cutout.CutoutImage.__subclasses__():
        if subclass.__name__ == cutout_class:
            class_instance = subclass

    # throw an error if we did not find anything
    if class_instance is None:
        raise ValueError(f"Chosen class does not exist: {cutout_class}")

    cut = class_instance(channel)
    if corners is not None:
        cut.corners_cut = corners
    cut.run_align_cutout()

    return cut.corners_cut


# run as main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, nargs="+", required=True,
                        help="The channels used for to cut out the images.")
    parser.add_argument("--cutout_class", type=str, required=True,
                        help="Name of the class used to perform the chamber cutout. Must be defined in a file of "
                             "midap.imcut and a subclass of midap.imcut.base_cutout.CutoutImage")
    args = parser.parse_args()

    # unpack the namespace
    main(**vars(args))
