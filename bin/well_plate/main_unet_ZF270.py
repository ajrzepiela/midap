from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np

import sys
sys.path.append('../../src')
from model import unet, unet_inference, unet_reduced, unet_reduced_inference
from unet_preprocessing import DataProcessor


# Load the data
path = '../../data/unet/well_plate/ZF270/'
proc = DataProcessor()

path_img_train = [
    path + 'train/img/ImprovedStackWell_frame33_OrigIm_cut1.tif',
    path + 'train/img/ImprovedStackWell_frame33_OrigIm_cut2.tif']
path_mask_train = [
    path + 'train/mask/ImprovedStackWell_frame33_MaskOnly_cut1.tif',
    path + 'train/mask/ImprovedStackWell_frame33_MaskOnly_cut2.tif']
path_mask_splitting_events = [
    path + 'train/mask_splitting/ImprovedStackWell_frame33_MaskSplitting_cut1.tif',
    path + 'train/mask_splitting/ImprovedStackWell_frame33_MaskSplitting_cut2.tif']
path_img_test = '../data/data_Johannes/unet/Well2_frame110.png'

# Preprocessing of data
X_train = []
y_train = []
weight_maps_train = []
splitting_cells_train = []
X_val = []
y_val = []
weight_maps_val = []
splitting_cells_val = []

for i in range(len(path_img_train)):
    (X_train_tmp,
     y_train_tmp,
     weight_maps_train_tmp,
     splitting_cells_train_tmp,
     ratio_cell_train_bin,
     X_val_tmp,
     y_val_tmp,
     weight_maps_val_tmp,
     splitting_cells_val_tmp,
     ratio_cell_bin_val) = proc.run(path_img_train[i],
                                    path_mask_train[i],
                                    path_mask_splitting_events[i])

    X_train.append(X_train_tmp)
    y_train.append(y_train_tmp)
    weight_maps_train.append(weight_maps_train_tmp)
    splitting_cells_train.append(splitting_cells_train_tmp)
    X_val.append(X_val_tmp)
    y_val.append(y_val_tmp)
    weight_maps_val.append(weight_maps_val_tmp)
    splitting_cells_val.append(splitting_cells_val_tmp)

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
weight_maps_train = np.concatenate(weight_maps_train)
splitting_cells_train = np.concatenate(splitting_cells_train)
X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)
weight_maps_val = np.concatenate(weight_maps_val)
splitting_cells_val = np.concatenate(splitting_cells_val)

# Train model
val_split = 0.3
batch_size = 2

sample_weight = splitting_cells_train
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
model.save_weights('../../model_weights/well_plate/model_weights_ZF270withWeighting.h5')
