import sys
sys.path.append('../src')

from unet_preprocessing import DataProcessor
from unet_prediction import ImagePadding
from model import unet, unet_inference, unet_reduced, unet_reduced_inference

import numpy as np
from skimage import io

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

path = '../data/data_Johannes/unet/'
proc = DataProcessor(path)

path_img_train = '../data/data_Johannes/unet/ImprovedStackWell_frame33_OrigIm_cutout.tif'
path_mask_train = '../data/data_Johannes/unet/ImprovedStackWell_frame33_MaskOnly_Edt_cutout.tif'
path_img_test = '../data/data_Johannes/Well2_frame110.png' #images/Well2ImprovedStack_all_frames/Well2ImprovedStack_frame_120.tif'

X_train, y_train, weight_maps_train, X_val, y_val, weight_maps_val = proc.run(path_img_train, path_mask_train)

print('train model')
val_split = 0.3
batch_size=2

model = unet_reduced(input_size = (X_train.shape[1], X_train.shape[2], 1), dropout = 0.5)
callback = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x=[X_train, weight_maps_train], y=y_train, epochs=1, validation_data=([X_val, weight_maps_val], y_val), batch_size=batch_size, callbacks=[callback])
model.save_weights('model_weights_well_w_distmap.h5')

pad = ImagePadding()
img_pad = pad.run_single_image(path_img_test)

model_pred = unet_reduced_inference(input_size = img_pad.shape[1:3] + (1,))
model_pred.set_weights(model.get_weights())
#model_pred.load_weights('model_weights_well.h5')
y_pred = model_pred.predict(img_pad)

seg = pad.undo_padding(y_pred)

# model_pred = unet_reduced_inference(input_size = (128, 128, 1))
# model_pred.set_weights(model.get_weights())
# #model_pred.load_weights('model_weights_well.h5')
# y_pred = model_pred.predict(X_test)
# np.save(path + 'prediction_patches.npy', y_pred)

# # generate prediction for whole image
# img_test = io.imread(path_img_test)
# #img_test = (img_test - np.mean(img_test))/(np.std(img_test))
# new_shape = (int(np.ceil(img_test.shape[0]/16)*16), int(np.ceil(img_test.shape[1]/16)*16))
# img_pad = np.zeros(new_shape)
# img_pad[:img_test.shape[0], :img_test.shape[1]] = img_test
# img_pad = img_pad.reshape((1,) + new_shape + (1,))

# model_pred = unet_reduced_inference(input_size = new_shape + (1,))
# model_pred.set_weights(model.get_weights())
# #model_pred.load_weights('model_weights_well.h5')
# y_pred = model_pred.predict(img_pad)
# np.save(path + 'prediction.npy', y_pred[0,:,:,0])
