"""
Correct segmentations
===============================

Correct a segmentation generated with Midap
"""
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
from pathlib import Path
from skimage import io
from typing import List, Tuple
from tqdm import tqdm

from midap.correction.napari_correction import Correction

def load_tif(path_img: str) -> Tuple[str, List[str]]:
    """
    Load tif file and split stack into single frames and saves
    frames as single images.
    :param path_img: Path to the tif stack.
    :return: Directory and file names of images.
    """

    img_stack = io.imread(path_img)
    num_frames = len(img_stack)

    directory = 'cut_im/'
    raw_filename = path_img.stem
    save_dir = path_img.parent.joinpath(raw_filename, directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    print('Splitting frames (image stack)...')
    for ix in tqdm(range(num_frames)):
        io.imsave(save_dir.joinpath(f"_{raw_filename}_frame{ix:03d}_cut.png"),
                      (img_stack[ix]*255).astype(np.uint8), check_contrast=False)
        
    files_cut_im = sorted(os.listdir(save_dir))

    return save_dir, files_cut_im


def load_h5(path_seg: str, thr: float = 0.9) -> Tuple[str, List[str]]:
    """
    Load h5 file and split file into single thresholded frames and saves
    frames as single images.
    :param path_seg: Path to the h5 file.
    :param thr: Threshold for image binarization.
    :return: Directory and file names of segmentations.
    """

    f = h5py.File(path_seg)
    key = list(f.keys())[0]
    dset = np.array(f[key])
    num_frames = len(dset)

    directory = 'seg_im/'
    raw_filename = path_seg.stem
    save_dir = path_seg.parent.joinpath(raw_filename, directory)

    save_dir.mkdir(parents=True, exist_ok=True)

    print('Splitting frames (segmentation)...')
    for ix in tqdm(range(num_frames)):
        io.imsave(save_dir.joinpath(f"_{raw_filename}_frame{ix:03d}_seg.png"),
                      255*(dset[ix][:,:,0] > thr).astype(np.uint8), check_contrast=False)
        
    files_seg_im = sorted(os.listdir(save_dir))

    return save_dir, files_seg_im



def get_file_names(path_img: str, path_seg: str) -> Tuple[str, str, List[str], List[str]]:
    """
    Get file names of all raw images and segmentations.
    :param path_img: Path to image file/folder.
    :param path_seg: Path to segmentation file/folder.
    :return: Directory and file names of images and segmentations.
    """

    path_img = Path(path_img)
    path_seg = Path(path_seg)
    
    img_is_dir = path_img.is_dir()
    seg_is_dir = path_seg.is_dir()

    if img_is_dir:
        files_cut_im = sorted(os.listdir(path_img))
        dir_cut_im = path_img
    elif path_img.suffix == '.tif':
        dir_cut_im, files_cut_im = load_tif(path_img)
    else:
        raise TypeError(f"Unsupported file type '{path_img.suffix}' "
                                f"Only directories and .tif-files are supported.")


    if seg_is_dir:
        files_seg_im = sorted(os.listdir(path_seg))
        dir_seg_im = path_seg
    elif path_seg.suffix == '.h5':
        dir_seg_im, files_seg_im = load_h5(path_seg)
    else:
        raise TypeError(f"Unsupported file type '{path_seg.suffix}' "
                                f"Only directories and .h5-files are supported.")

    return dir_cut_im, dir_seg_im, files_cut_im, files_seg_im

def main() -> None:
    """
    Main function to run the segmentation correction with napari.
    """

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img', type=str, required=True, help='Path to raw image folder.')
    parser.add_argument('--path_seg', type=str, required=True, help='Path to segmentation folder.')
    args = parser.parse_args()

    # get file names
    dir_cut_im, dir_seg_im, files_cut_im, files_seg_im = get_file_names(args.path_img, args.path_seg)

    # plot first time frame
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    callback = Correction(ax, dir_cut_im, dir_seg_im, files_cut_im, files_seg_im)
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
