import ipywidgets as widgets
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.segmentation import mark_boundaries
from skimage.measure import label

import subprocess

def button_show_files():
    dropdown_files = widgets.Dropdown(options=os.listdir())

    return dropdown_files

def button_show_images(dropdown_files):
    button = widgets.Button(description="Show images")
    output = widgets.Output()

    display(button, output)
    
    def on_button_clicked(b):
        raw_images = glob(dropdown_files.value+'/*raw.tif')
        print(raw_images)
        seg_images = [r.replace('raw', 'seg') for r in raw_images]
        with output:
            num_subp = int(np.ceil(np.sqrt(len(raw_images))))
            plt.figure(figsize=(10,15))
            i_fig = 1
            for r, s in zip(raw_images, seg_images):
                #plt.subplot(num_subp,num_subp,i_fig)
                plt.figure();
                img = io.imread(r);
                seg = io.imread(s);
                io.imshow(mark_boundaries(img, label(seg)));
                plt.title(r);
                i_fig+=1
            plt.show()

    return button.on_click(on_button_clicked)

def button_start_training():
    button_train = widgets.Button(description="Start training")
    display(button_train)

    def on_button_clicked(b):
        rc = subprocess.call("./run_training.sh", shell=True)


    return button_train.on_click(on_button_clicked)