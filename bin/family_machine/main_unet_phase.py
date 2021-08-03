from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np

import sys
sys.path.append('../src')
from model import unet, unet_inference, unet_reduced, unet_reduced_inference
from unet_preprocessing import DataProcessor

# Load data
path = '../data/unet/Phase/'
proc = DataProcessor()

path_img_train = [
    path + 'train/img/CWRt00100xy001c1.tif',
    path + 'train/img/CWRt00200xy001c1.tif']
path_mask_train = [
    path + 'train/mask/CWRt00100xy001c1_bin.tif',
    path + 'train/mask/CWRt00200xy001c1_bin.tif']

# Preprocessing of data
X_train = []
y_train = []
weight_maps_train = []
ratio_cell_train = []
X_val = []
y_val = []
weight_maps_val = []
ratio_cell_val = []

for i in range(len(path_img_train)):
    X_train_tmp, y_train_tmp, weight_maps_train_tmp, ratio_cell_train_tmp, X_val_tmp, y_val_tmp, weight_maps_val_tmp, ratio_cell_val_tmp = \
        proc.run(path_img_train[i], path_mask_train[i])

    X_train.append(X_train_tmp)
    y_train.append(y_train_tmp)
    weight_maps_train.append(weight_maps_train_tmp)
    ratio_cell_train.append(ratio_cell_train_tmp)

    X_val.append(X_val_tmp)
    y_val.append(y_val_tmp)
    weight_maps_val.append(weight_maps_val_tmp)
    ratio_cell_val.append(ratio_cell_val_tmp)

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
weight_maps_train = np.concatenate(weight_maps_train)
ratio_cell_train = np.concatenate(ratio_cell_train)

X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)
weight_maps_val = np.concatenate(weight_maps_val)
ratio_cell_val = np.concatenate(ratio_cell_val)

# Train model
val_split = 0.3
batch_size = 2

sample_weight = ratio_cell_train
model = unet(input_size=(X_train.shape[1], X_train.shape[2], 1), dropout=0.5)
callback = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x=[X_train,
             weight_maps_train],
          y=y_train,
          sample_weight=sample_weight,
          epochs=50,
          validation_data=([X_val,
                            weight_maps_val],
                           y_val),
          batch_size=batch_size,
          callbacks=[callback],
          shuffle=True)
model.save_weights('../model_weights/model_weights_phase.h5')
