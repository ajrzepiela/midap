import sys
sys.path.append('../../src')

from unet_preprocessing import DataProcessor
from model import unet, unet_wo_weighting, unet_inference, unet_reduced, unet_reduced_inference

import numpy as np
from skimage import io

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

# Load data
path = '../../data/unet/mother_machine/Vibrio_natriegens/'
proc = DataProcessor()

path_img_train = path + 'train/img/'
path_mask_train = path + 'train/mask/'
path_img_test = path + 'test/img/'

# Data augmentation
X_train, y_train, X_val, y_val = proc.run_mother_machine(path_img_train, path_mask_train)

X_train = np.expand_dims(X_train, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)

# Train model
print('train model')
val_split = 0.3
batch_size=2

model = unet_wo_weighting(input_size = (X_train.shape[1], X_train.shape[2], 1), dropout = 0.5)
callback = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x=X_train, y=y_train, epochs=50, validation_data=(X_val, y_val), batch_size=batch_size, callbacks=[callback], shuffle=True)
model.save_weights('../../model_weights/model_weights_vibrio_natriegens.h5')
