"""
Correct segmentations
===============================

Correct a segmentation generated with Midap
"""
import argparse
import skimage.io as io
from skimage import measure
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import re
import os
import numpy as np
import pdb

from midap.correction.napari_correction import Correction


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img', help='path to images')
    parser.add_argument('--path_seg', help='path to segmentations')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # get file names
    files_cut_im = np.sort(os.listdir(args.path_img))
    files_seg_im = np.sort(os.listdir(args.path_seg))

    # plot first time frame
    frame_cut = re.findall('_frame[0-9][0-9][0-9]_', files_cut_im[0])[0]
    ix_seg = np.where([frame_cut in fs for fs in files_seg_im])[0][0]

    cut_im = io.imread(args.path_img + '/' + files_cut_im[0])
    seg_im = io.imread(args.path_seg + '/' + files_seg_im[ix_seg])

    overl = mark_boundaries(cut_im, seg_im, color=(1, 0, 0))

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    im1 = ax.imshow(overl)
    ax.set_title(str(frame_cut))

    # include buttons
    callback = Correction(ax, im1, args.path_img,
                          args.path_seg, files_cut_im, files_seg_im)
    axprev = fig.add_axes([0.55, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.66, 0.05, 0.1, 0.075])
    axnapari = fig.add_axes([0.77, 0.05, 0.13, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next_frame)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev_frame)
    bnapari = Button(axnapari, 'Correction')
    bnapari.on_clicked(callback.correct_seg)

    plt.show()


if __name__ == '__main__':
    main()
