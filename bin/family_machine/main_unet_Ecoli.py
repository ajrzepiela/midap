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
data = np.load(path + 'train/training_data.npz')

X_train = data['X_train']
y_train = data['y_train']
weight_maps_train = data['weight_maps_train']
ratio_cell_train = data['ratio_cell_train']
X_val = data['X_val']
y_val = data['y_val']
weight_maps_val = data['weight_maps_val']
ratio_cell_val = data['ratio_cell_val']

# Train model
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
model.save_weights('../../model_weights/family_machine/model_weights_Ecoli.h5')
