import skimage.io as io
from skimage.measure import label

import numpy as np
import os

from model import unet_inference

import matplotlib
import matplotlib.pyplot as plt

class SegmentationPredictor():
    
    '''
    Prediction of segmentation for specific cell type based on the 
    given model weights.
    '''

    def __init__(self, model_weights, div = 16):
        self.model_weights = model_weights # model weights trained for specific cell type
        self.div = div #divisor

    def predict_single_image(self, path_img):
        
        '''
        Generate prediction for single image.
        path_img: path to raw image which should
        be segmented.
        '''
        
        self.img = self.scale_pixel_vals(io.imread(path_img))
        img_pad = self.pad_image(self.img)

        model_pred = unet_inference(input_size = img_pad.shape[1:3] + (1,))
        model_pred.load_weights(self.model_weights)
        y_pred = model_pred.predict(img_pad)
        self.seg = (self.undo_padding(y_pred) > 0.5).astype(int)
        self.seg_label = label(self.seg)
        
    def save_segmentation_image(self, path_seg, img):
        
        '''
        Save generated segmentation depending on
        datatype of parsed image (numpy array or
        matplotlib figure).
        '''
        
        if type(img) == matplotlib.figure.Figure:
            plt.savefig(path_seg, dpi = 900)
        elif type(img) == np.ndarray:
            io.imsave(path_seg, img, check_contrast=False)
        
    def generate_segmentation_overlay(self):
        
        '''
        Generate overlay of raw image and
        generated segmentation.
        '''
        
        self.fig = plt.figure(figsize = (7,7))
        plt.imshow(self.img)
        plt.contour(self.seg, [0.5], colors = 'r', linewidths = 0.2)
        plt.xticks([])
        plt.yticks([])

        
#     def run_single_image(self, path_img, img_name, path_seg, seg_name=None):
#         print('Image padding')
#         img = self.scale_pixel_vals(io.imread(path_img + img_name))
#         img_pad = self.pad_image(img)
#         print(img_pad.shape)

#         print('Image segmentation')
#         model_pred = unet_inference(input_size = img_pad.shape[1:3] + (1,))
#         model_pred.load_weights(self.model_weights)
#         y_pred = model_pred.predict(img_pad)
#         seg = (self.undo_padding(y_pred) > 0.5).astype(int)
#         seg_label = label(seg)

#         print('Segmentation storage')
#         if seg_name:
#             io.imsave(path_seg + seg_name, seg_label, check_contrast=False)
#         else:
#             io.imsave(path_seg + img_name[:-4] + 'seg.png', seg_label, check_contrast=False)


    def run_image_stack(self, path_pos, path_cut, path_seg):
        
        '''
        Run prediction for whole image stack.
        '''
        
        print('Image padding')
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
        y_preds = model_pred.predict(imgs_pad[:10])

        print('Segmentation storage')
        if not os.path.exists(path_pos + path_seg):
            os.makedirs(path_pos + path_seg)

        for i, y in enumerate(y_preds):
            seg = (self.undo_padding_stack(y) > 0.5).astype(int)
            seg_label = label(seg)
            io.imsave(path_pos + path_seg + path_imgs[i][:-7] + 'seg.png', seg_label, check_contrast=False)


    def scale_pixel_vals(self, img):
        
        '''
        Scale pixel values to values between 0 and 1.
        '''
        
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
