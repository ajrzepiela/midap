"""
Correct segmentations
===============================

Correct a segmentation generated with Midap
"""
import argparse
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import re
import os
import numpy as np
import napari
from napari.settings import SETTINGS

import pdb

SETTINGS.application.ipy_interactive = False

class Correction:
    frame = 0

    def __init__(self, path_img, path_seg, files_cut_im, files_seg_im, obj, obj1, ax):
        self.path_img = path_img
        self.path_seg = path_seg
        self.files_cut_im = files_cut_im
        self.files_seg_im = files_seg_im
        self.obj = obj
        self.obj1 = obj1
        self.ax = ax


    def open_napari(self, event):
        #initialise viewer with raw image and segmentation
        frame_cut = re.findall('_frame[0-9][0-9][0-9]_', self.files_cut_im[self.frame])[0]
        ix_seg = np.where([frame_cut in fs for fs in self.files_seg_im])[0][0]

        cut_im = io.imread(self.path_img + '/' + self.files_cut_im[self.frame])
        seg_im = io.imread(self.path_seg + '/' + self.files_seg_im[ix_seg])

        viewer = napari.Viewer()
        viewer.add_image(cut_im)
        label_layer = viewer.add_labels(seg_im)

        napari.run()

        # save corrected segmentation image
        edited_labels = label_layer.data
        viewer.close()

        # name = '.'.join(args.path_seg.split('.')[:-1])
        # ext = args.path_seg.split('.')[-1]
        # path_corr = name + '_corr.' + ext
        # io.imsave(path_corr, edited_labels)
        

    def next_frame(self, event):
        # load data of next time frame
        self.frame+=1
        frame_cut = re.findall('_frame[0-9][0-9][0-9]_', self.files_cut_im[self.frame])[0]
        ix_seg = np.where([frame_cut in fs for fs in self.files_seg_im])[0][0]

        cut_im = io.imread(self.path_img + '/' + self.files_cut_im[self.frame])
        seg_im = io.imread(self.path_seg + '/' + self.files_seg_im[ix_seg])
        seg_im_bin = np.ma.masked_where(seg_im == 0, seg_im)

        # update figure
        self.obj.set_data(cut_im)
        self.obj1.set_data(seg_im_bin)
        self.ax.set_title(str(frame_cut))
        plt.draw()

    def prev_frame(self, event):
        # load data of prev frame
        self.frame-=1
        frame_cut = re.findall('_frame[0-9][0-9][0-9]_', self.files_cut_im[self.frame])[0]
        ix_seg = np.where([frame_cut in fs for fs in self.files_seg_im])[0][0]

        cut_im = io.imread(self.path_img + '/' + self.files_cut_im[self.frame])
        seg_im = io.imread(self.path_seg + '/' + self.files_seg_im[ix_seg])
        seg_im_bin = np.ma.masked_where(seg_im == 0, seg_im)

        # update figure
        self.obj.set_data(cut_im)
        self.obj1.set_data(seg_im_bin)
        self.ax.set_title(str(frame_cut))
        plt.draw()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img', help='path to images')
    parser.add_argument('--path_seg', help='path to segmentations')
    args = parser.parse_args()
    return args

#def load_data():

def main():
    #path_img = '../example_data/Pos57/TXRED/cut_im'
    #path_seg = '../example_data/Pos57/TXRED/seg_im'

    args = get_args()

    # get folder with imgs and segms

    # compare frame numbers
    files_cut_im = np.sort(os.listdir(args.path_img))
    files_seg_im = np.sort(os.listdir(args.path_seg))

    #plot first time frame
    frame_cut = re.findall('_frame[0-9][0-9][0-9]_', files_cut_im[0])[0]
    ix_seg = np.where([frame_cut in fs for fs in files_seg_im])[0][0]

    cut_im = io.imread(args.path_img + '/' + files_cut_im[0])
    seg_im = io.imread(args.path_seg + '/' + files_seg_im[ix_seg])
    seg_im_bin = (seg_im > 0).astype(int)
    seg_im_bin = np.ma.masked_where(seg_im == 0, seg_im)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    l1 = ax.imshow(cut_im, cmap='gray')
    l2 = ax.imshow(seg_im_bin, alpha = 0.5)
    ax.set_title(str(frame_cut))

    callback = Correction(args.path_img, args.path_seg, files_cut_im, files_seg_im, l1, l2, ax)
    axprev = fig.add_axes([0.55, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.66, 0.05, 0.1, 0.075])
    axnapari = fig.add_axes([0.77, 0.05, 0.13, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next_frame)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev_frame)
    bprev = Button(axnapari, 'Correction')
    bprev.on_clicked(callback.open_napari)

    plt.show()

if __name__ == '__main__':
    main()
    

    