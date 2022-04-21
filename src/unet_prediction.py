import skimage.io as io
from skimage.measure import label, regionprops
from skimage.morphology import area_closing
#from scipy.ndimage.measurements import label
import scipy.ndimage as ndi

import numpy as np
import os
import re

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons
from matplotlib_widget import MyRadioButtons

from model import unet_inference

from skimage.filters import sobel
from skimage.segmentation import watershed

class SegmentationPredictor():

    def __init__(self, path_model_weights, postprocessing, div = 16, connectivity = 1):
        self.path_model_weights = path_model_weights
        self.postprocessing = postprocessing
        self.div = div # divisor
        self.connectivity = connectivity

    def segment_region_based(self, img, min_val = 40, max_val = 50):
        elevation_map = sobel(img)
        markers = np.zeros_like(img)
        markers[img < min_val] = 1
        markers[img > max_val] = 2
        segmentation = watershed(elevation_map, markers)
        return (segmentation <= 1).astype(int)

    def select_weights(self, path_pos, path_cut, path_seg):
        print('Select weights')
        list_files = np.sort(os.listdir(path_pos + path_cut))
        ix_half = int(len(list_files)/2)
        path_img = list_files[ix_half]

        img = self.scale_pixel_vals(io.imread(path_pos + path_cut + path_img))
        img_pad = self.pad_image(img)

        # try watershed segmentation as classical segmentation method
        watershed_seg = self.segment_region_based(img, 0.16, 0.19)
        
        # compute sample segmentations for all stored weights
        model_weights = os.listdir(self.path_model_weights)
        segs = [watershed_seg]
        for m in model_weights:
            model_pred = unet_inference(input_size = img_pad.shape[1:3] + (1,))
            model_pred.load_weights(self.path_model_weights+m)
            y_pred = model_pred.predict(img_pad)
            seg = (self.undo_padding(y_pred) > 0.5).astype(int)
            segs.append(seg)
        
        # display different segmentation methods (watershed + NN trained for different cell types)
        labels = ['watershed']
        labels += [mw.split('.')[0].split('_')[-1] for mw in model_weights]
        num_subplots = int(np.ceil(np.sqrt(len(segs))))
        plt.figure(figsize = (10,10))
        for i,s in enumerate(segs):
            plt.subplot(num_subplots,num_subplots,i+1)
            plt.imshow(img)
            plt.contour(s, [0.5], colors = 'r', linewidths = 0.5)
            if i == 0:
                plt.title('watershed')
            else:
                plt.title('model trained for ' + labels[i])
            plt.xticks([])
            plt.yticks([])
        plt.suptitle('Select model weights for channel: ' + path_seg.split('/')[1])
        rax = plt.axes([0.3, 0.01, 0.3, 0.08])
        #visibility = [False for i in range(len(segs))]
        check = RadioButtons(rax, labels)

        plt.show()

        #extract selected segmentation method from output of RadioButton
        if check.value_selected == 'watershed':
            self.model_weights = 'watershed'
        else:
            ix_model_weights = np.where([check.value_selected == l for l in labels])[0][0]
            sel_model_weights = model_weights[ix_model_weights - 1]
            self.model_weights = self.path_model_weights + sel_model_weights


    def run_single_image(self, path_pos, path_cut, path_seg, path_img):
        print('Image padding')
        img = self.scale_pixel_vals(io.imread(path_pos + path_cut + path_img))
        img_pad = self.pad_image(img)

        print('Image segmentation')
        model_pred = unet_inference(input_size = img_pad.shape[1:3] + (1,))
        model_pred.load_weights(self.model_weights)
        y_pred = model_pred.predict(img_pad)
        seg = (self.undo_padding(y_pred) > 0.5).astype(int)
        seg_label = label(seg, connectivity=self.connectivity)

        print('Segmentation storage')
        io.imsave(path_pos + path_seg + path_img.replace('_cut.tif', '').replace('.tif', '') + '_seg.png', seg_label.astype('uint16'), check_contrast=False)


    def run_image_stack(self, path_pos, path_cut, path_seg, path_seg_track, model_weights):
        path_imgs = np.sort(os.listdir(path_pos + path_cut))

        if model_weights == 'watershed':
            print('Image segmentation and storage')
            segs = []
            cuts = []
            for p in path_imgs:
                img = self.scale_pixel_vals(io.imread(path_pos + path_cut + p))
                seg = self.segment_region_based(img, 0.16, 0.19)
                cuts.append(img)
                if self.postprocessing == True:
                    seg = self.postprocess_seg(seg)
                segs.append(seg)
                seg_label  = label(seg, connectivity=self.connectivity)

                io.imsave(path_pos + path_seg + p.replace('_cut.tif', '').replace('_cut.png', '').replace('.tif', '') + '_seg.png', seg_label.astype('uint16'), check_contrast=False)
            io.imsave(path_pos + path_seg_track + path_imgs[0].replace('_cut.tif', '').replace('_cut.png', '').replace('.tif', '') + '_full_stack_seg_bin.png', np.array(segs))
        
        else:
            print('Image padding')
            imgs_pad = []
            for p in path_imgs:
                img = self.scale_pixel_vals(io.imread(path_pos + path_cut + p))
                img_pad = self.pad_image(img)
                imgs_pad.append(img_pad)
            imgs_pad = np.concatenate(imgs_pad)

            print('Image segmentation')
            model_pred = unet_inference(input_size = imgs_pad.shape[1:3] + (1,))
            model_pred.load_weights(model_weights)
            y_preds = model_pred.predict(imgs_pad, batch_size = 1, verbose = 1)

            print('Segmentation storage')
            segs = []
            for i, y in enumerate(y_preds):
                seg = (self.undo_padding_stack(y) > 0.5).astype(int)
                if self.postprocessing == True:
                    seg = self.postprocess_seg(seg)
                segs.append(seg)
                seg_label  = label(seg, connectivity=self.connectivity)

                io.imsave(path_pos + path_seg + path_imgs[i].replace('_cut.tif', '').replace('_cut.png', '').replace('.tif', '') + '_seg.tiff', seg_label.astype('uint16'), check_contrast=False)

            io.imsave(path_pos + path_seg_track + path_imgs[0].replace('_cut.tif', '').replace('_cut.png', '').replace('.tif', '') + '_full_stack_cut.tiff', np.array(imgs_pad).astype(float))
            io.imsave(path_pos + path_seg_track + path_imgs[0].replace('_cut.tif', '').replace('_cut.png', '').replace('.tif', '') + '_full_stack_seg_bin.tiff', np.array(segs))
            io.imsave(path_pos + path_seg_track + path_imgs[0].replace('_cut.tif', '').replace('_cut.png', '').replace('.tif', '') + '_full_stack_seg_prob.tiff', y_preds.astype(float))

    def postprocess_seg(self, seg):
	# remove small and big particels which are not cells
        label_objects = label(seg, connectivity = self.connectivity)
        sizes = np.bincount(label_objects.ravel())
        reg = regionprops(label_objects)
        areas = [r.area for r in reg]
        #min_size, max_size = np.quantile(areas, [0.01, 1.])
        min_size = np.quantile(areas, [0.01])
        #mask_sizes = (sizes > min_size)&(sizes < max_size)
        mask_sizes = (sizes > min_size)
        mask_sizes[0] = 0
        img_filt = (mask_sizes[label_objects] > 0).astype(int)
	
	# close small holes
        img_closed = area_closing(img_filt)
        return img_closed

    def scale_pixel_vals(self, img):
        img = np.array(img)
        return ((img - img.min())/(img.max() - img.min()))


    def pad_image(self, img):
        new_shape = (int(np.ceil(img.shape[0]/self.div)*self.div), int(np.ceil(img.shape[1]/self.div)*self.div))
        img_pad = np.zeros(new_shape)

        self.row_shape = img.shape[0]
        self.col_shape = img.shape[1]

        img_pad[:self.row_shape, :self.col_shape] = img
        img_pad[self.row_shape:, :self.col_shape] = img[(img.shape[0] - (img_pad.shape[0] - img.shape[0])):, :][::-1] #pad mirrored image
        img_pad[:self.row_shape, self.col_shape:] = np.fliplr(np.flipud(img[:, (img.shape[1] - (img_pad.shape[1] - img.shape[1])):][::-1])) #pad mirrored image
        img_pad = img_pad.reshape((1,) + new_shape + (1,))

        return img_pad


    def undo_padding(self, img_pad):
        img_unpad = img_pad[0,:self.row_shape, :self.col_shape,0]
        return img_unpad


    def undo_padding_stack(self, img_pad):
        img_unpad = img_pad[:self.row_shape, :self.col_shape,0]
        return img_unpad
