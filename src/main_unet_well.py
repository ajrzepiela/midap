import sys
sys.path.append('../src')

from unet_preprocessing import Preprocessing
from model import unet, unet_inference, unet_reduced, unet_reduced_inference

import numpy as np
from skimage import io

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

path = '../data/data_Johannes/unet/'
proc = Preprocessing(path, shift = 64)

path_img_train = '../data/data_Johannes/unet/ImprovedStackWell_frame33_OrigIm_cutout.tif'
path_mask_train = '../data/data_Johannes/unet/ImprovedStackWell_frame33_MaskOnly_Edt_cutout.tif'
path_img_test = '../data/data_Johannes/Well2_frame_110.png' #images/Well2ImprovedStack_all_frames/Well2ImprovedStack_frame_120.tif'

print('data augmentation')
X_train, y_train, weight_maps_train = proc.build_train_patches(path_img_train, path_mask_train, num=200)#200
X_val, y_val, weight_maps_val = proc.build_val_patches(path_img_train, path_mask_train, num=100)#100
X_test = proc.build_test_patches(path_img_test)

print('train model')
val_split = 0.3
batch_size=2

model = unet_reduced(input_size = (128, 128, 1), dropout = 0.8)
callback = EarlyStopping(monitor='val_loss', patience=3)
model.fit(x=[X_train, weight_maps_train], y=y_train, epochs=1, validation_data=([X_val, weight_maps_val], y_val), batch_size=batch_size, callbacks=[callback])
model.save_weights('model_weights_well_w_distmap.h5')

model_pred = unet_reduced_inference(input_size = (128, 128, 1))
model_pred.set_weights(model.get_weights())
#model_pred.load_weights('model_weights_well.h5')
y_pred = model_pred.predict(X_test)
np.save(path + 'prediction_patches.npy', y_pred)

# generate prediction for whole image
img_test = io.imread(path_img_test)
img_test = (img_test - np.mean(img_test))/(np.std(img_test))
new_shape = (int(np.ceil(img_test.shape[0]/16)*16), int(np.ceil(img_test.shape[1]/16)*16))
img_pad = np.zeros(new_shape)
img_pad[:img_test.shape[0], :img_test.shape[1]] = img_test
img_pad = img_pad.reshape((1,) + new_shape + (1,))

model_pred = unet_reduced_inference(input_size = new_shape + (1,))
model_pred.set_weights(model.get_weights())
#model_pred.load_weights('model_weights_well.h5')
y_pred = model_pred.predict(img_pad)
np.save(path + 'prediction.npy', y_pred[0,:,:,0])
