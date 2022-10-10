import argparse
import os

# import to get all subclasses
from midap.imcut import *
from midap.imcut import base_cutout

parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=str, nargs="+", required=True,
                    help="The channels used for to cut out the images.")
parser.add_argument("--cutout_class", type=str, required=True,
                    help="Name of the class used to perform the chamber cutout. Must be defined in a file of "
                         "midap.imcut and a subclass of midap.imcut.base_cutout.CutoutImage")
args = parser.parse_args()

# get the right subclass
cutout_class = None
for subclass in base_cutout.CutoutImage.__subclasses__():
    if subclass.__name__ == args.cutout_class:
        cutout_class = subclass

# throw an error if we did not find anything
if cutout_class == None:
    raise ValueError(f"Chosen class does not exist: {args.cutout_class}")

cut = cutout_class(args.channel, min_x_range = 700, max_x_range = 1600, min_y_range = 500, max_y_range = 1600)
cut.run_align_cutout()
