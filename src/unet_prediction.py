import skimage.io as io
from skimage.measure import label

import numpy as np
import os

from model import unet_inference

class SegmentationPredictor():

    def __init__(self, model_weights, div = 16):
        self.model_weights = model_weights
        self.div = div # divisor

    def run_single_image(self, path_pos, path_cut, path_seg, path_img):
        print('Image padding')
        img = self.scale_pixel_vals(io.imread(path_pos + path_cut + path_img))
        img_pad = self.pad_image(img)

        print('Image segmentation')
        model_pred = unet_inference(input_size = img_pad.shape[1:3] + (1,))
        model_pred.load_weights(self.model_weights)
        y_pred = model_pred.predict(img_pad)
        seg = (self.undo_padding(y_pred) > 0.5).astype(int)
        seg_label = label(seg)

        print('Segmentation storage')
        io.imsave(path_pos + path_seg + path_img[:-7] + 'seg.png', seg_label, check_contrast=False)


    def run_image_stack(self, path_pos, path_cut, path_seg):
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
        print(imgs_pad.shape)
        y_preds = model_pred.predict(imgs_pad, batch_size = 1, verbose = 1)

        print('Segmentation storage')
        if not os.path.exists(path_pos + path_seg):
            os.makedirs(path_pos + path_seg)

        for i, y in enumerate(y_preds):
            seg = (self.undo_padding_stack(y) > 0.5).astype(int)
            seg_label = label(seg)
            io.imsave(path_pos + path_seg + path_imgs[i][:-4] + '_seg.png', seg_label, check_contrast=False)


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