
from pathlib import Path
import os
import numpy as np
import shutil
import glob

from midap.utils import get_inheritors
from midap.segmentation import *
from midap.segmentation import base_segmentator
from midap.apps import segment_cells

segmentation_class = 'OmniSegmentation'

class_instance = None
for subclass in get_inheritors(base_segmentator.SegmentationPredictor):
    if subclass.__name__ == segmentation_class:
        class_instance = subclass

path_model_weights = "/Users/franziskaoschmann/Documents/midap/model_weights/model_weights_omni/"
postprocessing = True
img_threshold = 255
clean_border = False

models_omni_ph = glob.glob('/Users/franziskaoschmann/Documents/midap/model_weights/model_weights_omni/*phase*')
models_omni_fluor = glob.glob('/Users/franziskaoschmann/Documents/midap/model_weights/model_weights_omni/*fluor*')

def run_all_models(path_channel, path_model_weights, postprocessing, network_name, img_threshold):

    # get the Predictor
    pred = class_instance(path_model_weights=path_model_weights, postprocessing=postprocessing,
                            model_weights=network_name, img_threshold=img_threshold)

    # run the stack if we want to
    pred.run_image_stack(path_channel, clean_border)

    shutil.move(path_channel + 'seg_im/', path_channel + '/seg_im_' + segmentation_class + '_' + network_name.split('/')[-1].split('.')[0])

# Caulobacter LargeFamily
path_data = '/Users/franziskaoschmann/Documents/midap/benchmark_data/'
path_channel_caulo_ph = path_data + 'Caulobacter_LargeFamilyMachine/pos7/PH/'
path_channel_caulo_mcherry = path_data + 'Caulobacter_LargeFamilyMachine/pos7/mCherry/'

for network_name in models_omni_ph:
    run_all_models(path_channel_caulo_ph, path_model_weights, postprocessing, network_name, img_threshold)

for network_name in models_omni_fluor:
    run_all_models(path_channel_caulo_mcherry, path_model_weights, postprocessing, network_name, img_threshold)



# Ecoli LargeFamily
path_channel_ecoli_ph = path_data + 'Ecoli_LargeFamilyMachine/pos1/PH/'
path_channel_ecoli_mcherry = path_data + 'Ecoli_LargeFamilyMachine/pos1/mCherry/'

for network_name in models_omni_ph:
    run_all_models(path_channel_ecoli_ph, path_model_weights, postprocessing, network_name, img_threshold)

for network_name in models_omni_fluor:
    run_all_models(path_channel_ecoli_mcherry, path_model_weights, postprocessing, network_name, img_threshold)



# Ecoli SmallFamily
path_channel_ecoli_small_ph = path_data + 'Ecoli_SmallFamilyMachine/pos4/PH/'
path_channel_ecoli_small_gfp = path_data + 'Ecoli_SmallFamilyMachine/pos4/GFP/'

for network_name in models_omni_ph:
    run_all_models(path_channel_ecoli_small_ph, path_model_weights, postprocessing, network_name, img_threshold)

for network_name in models_omni_fluor:
    run_all_models(path_channel_ecoli_small_gfp, path_model_weights, postprocessing, network_name, img_threshold)



# Vcyc LargeFamily
path_channel_vcyc_ph = path_data + 'Vcyc_LargeFamilyMachine/pos13/PH/'
path_channel_vcyc_txred = path_data + 'Vcyc_LargeMachine/pos13/TxRed/'

for network_name in models_omni_ph:
    run_all_models(path_channel_vcyc_ph, path_model_weights, postprocessing, network_name, img_threshold)

for network_name in models_omni_fluor:
    run_all_models(path_channel_vcyc_txred, path_model_weights, postprocessing, network_name, img_threshold)