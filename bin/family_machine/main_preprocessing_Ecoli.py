from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np

import sys
sys.path.append('../../src')
from model import unet, unet_inference, unet_reduced, unet_reduced_inference
from unet_preprocessing import DataProcessor

# Load data
path = '../../data/unet/family_machine/Ecoli/'
proc = DataProcessor()

path_img_train = path + 'train/img/E_coli_train_1.tif'
path_mask_train = path + 'train/mask/segmented2.tif'

# Preprocessing of data
(X_train,
y_train,
weight_maps_train,
ratio_cell_train,
X_val,
y_val,
weight_maps_val,
ratio_cell_val) = proc.run(path_img_train,
                            path_mask_train)

# Save generated training data
np.savez_compressed(path + 'train/training_data_new_w.npz', X_train = X_train,
y_train = y_train,
weight_maps_train = weight_maps_train,
ratio_cell_train = ratio_cell_train,
X_val = X_val,
y_val = y_val,
weight_maps_val = weight_maps_val,
ratio_cell_val = ratio_cell_val)

