"""
Correct segmentations
===============================

Correct a segmentation generated with Midap
"""
import argparse
import skimage.io as io
import napari
from napari.settings import SETTINGS

SETTINGS.application.ipy_interactive = False

def load_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img', help='path to image')
    parser.add_argument('--path_seg', help='path to segmentation')
    args = parser.parse_args()
    return args

def main():
    args = load_data()

    # load the image and segment it
    image = io.imread(args.path_img)
    label_image = io.imread(args.path_seg)

    # initialise viewer with raw image and segmentation
    viewer = napari.Viewer()
    viewer.add_image(image)
    label_layer = viewer.add_labels(label_image)

    napari.run()

    # save corrected segmentation image
    edited_labels = label_layer.data

    name = '.'.join(args.path_seg.split('.')[:-1])
    ext = args.path_seg.split('.')[-1]
    path_corr = name + '_corr.' + ext
    io.imsave(path_corr, edited_labels)

if __name__ == '__main__':
    main()
    

    