import skimage.io as io
from skimage.measure import label
#from scipy.ndimage.measurements import label
import scipy.ndimage as ndi

import numpy as np
import os
import re

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons
from matplotlib_widget import MyRadioButtons

from model import unet_inference

class SegmentationPredictor():

    def __init__(self, postprocessing, div = 16, connectivity = 1):
        #self.model_weights = model_weights
        self.postprocessing = postprocessing
        self.div = div # divisor
        self.connectivity = connectivity

    def select_weights(self, path_pos, path_cut, path_seg):
        print('Select weights')
        path_img = np.sort(os.listdir(path_pos + path_cut))[-1]

        img = self.scale_pixel_vals(io.imread(path_pos + path_cut + path_img))
        img_pad = self.pad_image(img)
        
        model_weights = os.listdir('../model_weights/')
        segs = []
        for m in model_weights:
            model_pred = unet_inference(input_size = img_pad.shape[1:3] + (1,))
            model_pred.load_weights('../model_weights/'+m)
            y_pred = model_pred.predict(img_pad)
            seg = (self.undo_padding(y_pred) > 0.5).astype(int)
            segs.append(seg)
        
        labels = [mw.split('.')[0].split('_')[-1] for mw in model_weights]
        num_subplots = int(np.ceil(np.sqrt(len(model_weights))))
        plt.figure(figsize = (10,10))
        for i,s in enumerate(segs):
            plt.subplot(num_subplots,num_subplots,i+1)
            plt.imshow(img)
            plt.contour(s, [0], colors = 'r', linewidths = 0.5)
            plt.title('model weights trained for ' + labels[i])
            plt.xticks([])
            plt.yticks([])
        plt.suptitle('Select model weights for channel: ' + path_seg.split('/')[1])
        rax = plt.axes([0.3, 0.01, 0.3, 0.08])
        #labels = ['model weights ' + str(i + 1) for i in range(len(model_weights))]
        visibility = [False for i in range(len(model_weights))]
        # check = CheckButtons(rax, labels)
        check = RadioButtons(rax, labels)
        # check = MyRadioButtons(rax, labels, ncol=2)

        # def func(label):
        #     index = labels.index(label)
        #     #lines[index].set_visible(not lines[index].get_visible())
        #     plt.draw()

        # check.on_clicked(func)

        plt.show()

        #ix_model_weights = np.where(check.get_status())[0][0]
        ix_model_weights = np.where([check.value_selected == l for l in labels])[0][0]
        sel_model_weights = model_weights[ix_model_weights]
        self.model_weights = '../model_weights/' + sel_model_weights


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
        #io.imsave(path_pos + path_seg + path_img[:-7] + 'seg.tif', seg_label, check_contrast=False)
        io.imsave(path_pos + path_seg + path_img.replace('_cut.tif', '').replace('.tif', '') + '_seg.tiff', seg_label, check_contrast=False)


    def run_image_stack(self, path_pos, path_cut, path_seg):
        print('Image padding')
        print(self.model_weights)
        path_imgs = np.sort(os.listdir(path_pos + path_cut))

        imgs_pad = []
        for p in path_imgs:
            img = self.scale_pixel_vals(io.imread(path_pos + path_cut + p))
            img_pad = self.pad_image(img)
            imgs_pad.append(img_pad)
        imgs_pad = np.concatenate(imgs_pad)

        print('Image segmentation')
        model_pred = unet_inference(input_size = imgs_pad.shape[1:3] + (1,))
        model_pred.load_weights(self.model_weights)
        print(imgs_pad.shape)
        y_preds = model_pred.predict(imgs_pad, batch_size = 1, verbose = 1)

        print('Segmentation storage')
        if not os.path.exists(path_pos + path_seg):
            os.makedirs(path_pos + path_seg)

        #segs = []
        for i, y in enumerate(y_preds):
            seg = (self.undo_padding_stack(y) > 0.5).astype(int)
            if self.postprocessing == True:
                clean_seg = self.postprocess_seg(seg)
                seg_label  = label(clean_seg, connectivity=self.connectivity)
                print(np.unique(seg_label))
            #io.imsave(path_pos + path_seg + path_imgs[i][:-7] + '_seg.tif', seg_label, check_contrast=False)
            else:
                seg_label  = label(seg, connectivity=self.connectivity)
                print(np.unique(seg_label))
            io.imsave(path_pos + path_seg + path_imgs[i].replace('_cut.tif', '').replace('_cut.png', '').replace('.tif', '') + '_seg.tiff', seg_label, check_contrast=False)
            #segs.append(seg_label)
        #print(np.array(segs).shape)
        #io.imsave(path_pos + '/' + path_imgs[i][:-11] + '_full_stack.tif', np.array(segs))
        #new_filename = path_pos + '/' + re.sub(r'frame\d+_cut.tif', 'full_stack_seg.tif', path_imgs[i])
        #new_filename = path_pos + '/' + re.sub(r'frame\d+.tif', 'full_stack_seg.tif', path_imgs[i])
        #print(new_filename)
        #io.imsave('test.tif', np.array(segs)) #np.array(segs)

    def postprocess_seg(self, seg, min_size = 6, max_size = 100): #100
        label_objects = label(seg, connectivity = self.connectivity)
        sizes = np.bincount(label_objects.ravel())
        mask_sizes = (sizes > min_size)&(sizes < max_size)
        mask_sizes[0] = 0
        filtered_image = (mask_sizes[label_objects] > 0).astype(int)
        # label_seg = label(seg, connectivity = 1)
        # props = regionprops(label_seg)
        # areas = np.array([p.area for p in props])
        # mask_areas = (areas > 6)&(areas < 50)
        # filtered_image = mask_areas[seg].astype(int)

        # label_objects, _ = ndi.label(seg.astype(int))
        # sizes = np.bincount(label_objects.ravel())
        # mask_sizes = (sizes > min_size)&(sizes < max_size)
        # mask_sizes[0] = 0
        # filtered_image = (mask_sizes[label_objects] > 0).astype(int)
        return filtered_image

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