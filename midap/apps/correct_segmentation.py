"""
Correct segmentations
===============================

Correct a segmentation generated with Midap
"""
import argparse
import os
from typing import Union

import click
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from midap.correction.napari_correction import Correction


def get_args() -> None:
    '''
    Get arguments provided over the command line.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img', type=str, help='path to images')
    parser.add_argument('--path_seg', help='path to segmentations')
    args = parser.parse_args()
    return args


@click.command()
@click.option('--path_img', help='path to images', type=str)
@click.option('--path_seg', help='path to segmentations', type=str)
def main(path_img: Union[str, bytes, os.PathLike], path_seg: Union[str, bytes, os.PathLike]) -> None:
    '''
    Main function to run the segmentation correction with napari.
    '''

    # get file names
    files_cut_im = sorted(os.listdir(path_img))
    files_seg_im = sorted(os.listdir(path_seg))

    # plot first time frame
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    callback = Correction(ax, path_img, path_seg, files_cut_im, files_seg_im)
    callback.load_img_seg(0)

    im1 = ax.imshow(callback.overl)
    ax.set_title(str(callback.cur_frame))

    # include buttons
    axprev = fig.add_axes([0.55, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.66, 0.05, 0.1, 0.075])
    axnapari = fig.add_axes([0.77, 0.05, 0.13, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda x: callback.next_frame(x, im1))
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda x: callback.prev_frame(x, im1))
    bnapari = Button(axnapari, 'Correction')
    bnapari.on_clicked(callback.correct_seg)

    plt.show()


if __name__ == '__main__':
    main()
