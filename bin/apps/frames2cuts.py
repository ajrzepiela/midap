import argparse
import sys
sys.path.append('../src')
from image_cutout import CutoutImage

parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=str, nargs="+", required=True,
                    help="The channels used for to cut out the images.")
args = parser.parse_args()

cut = CutoutImage(args.channel, min_x_range = 700, max_x_range = 1600, min_y_range = 500, max_y_range = 1600)
cut.run_align_cutout()
