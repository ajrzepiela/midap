import sys
sys.path.append('../src')

from unet_prediction import SegmentationPredictor

pred = SegmentationPredictor('model_weights_mCherry.h5')
path_pos = '/Users/franziskaoschmann/Documents/ackermann-bacteria-segmentation/data/data_Glen/toy_R2/pos33/'
path_cut = 'cut/mCherry/'
path_seg = 'seg2/mCherry/'
pred.run_image_stack(path_pos, path_cut, path_seg)