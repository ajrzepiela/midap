import sys
sys.path.append('../src')

from unet_prediction import SegmentationPredictor

pred = SegmentationPredictor('model_weights_well.h5')
path_pos = '/Users/franziskaoschmann/Documents/ackermann-bacteria-segmentation/data/data_Johannes/'
path_cut = 'unet/'
path_seg = 'unet/'
path_img = 'Well2ImprovedStack_frame119.png'
new_path_img = 'Well2ImprovedStack_frame119_seg.png'
pred.run_single_image(path_pos, path_cut, path_seg, path_img, new_path_img)