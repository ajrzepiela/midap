import argparse

import sys
sys.path.append('../src')

from unet_prediction import SegmentationPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--path_pos")
parser.add_argument("--channel")
args = parser.parse_args()

pred = SegmentationPredictor('../model_weights/model_weights_'+args.channel+'.h5')
path_cut = '/GFP/xy1/phase/'
path_seg = '/GFP/seg_im/'
pred.run_image_stack(args.path_pos, path_cut, path_seg)
