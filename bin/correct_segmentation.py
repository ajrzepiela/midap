"""
Correct segmentations
===============================

Correct a segmentation generated with Midap
"""
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, square, remove_small_objects
import skimage.io as io
import napari
from napari.settings import SETTINGS

SETTINGS.application.ipy_interactive = False


def main():
    path_img = '../../../ackermann-bacteria-segmentation/data/unet/family_machine/CB15-flgH/Caulobacter-flgH-Raw.tif'
    path_seg = '../../../ackermann-bacteria-segmentation/data/unet/family_machine/CB15-flgH/Caulobacter-flgH-segmented.tif'

    # load the image and segment it
    image = io.imread(path_img)
    label_image = io.imread(path_seg)

    # initialise viewer with raw image and segmentation
    viewer = napari.Viewer()
    viewer.add_image(image)
    label_layer = viewer.add_labels(label_image)

    napari.run()

    # save corrected segmentation image
    edited_labels = label_layer.data

    name = '.'.join(path_seg.split('.')[:-1])
    ext = path_seg.split('.')[-1]
    path_corr = name + '_corr.' + ext
    io.imsave(path_corr, edited_labels)

if __name__ == '__main__':
    main()
    

    