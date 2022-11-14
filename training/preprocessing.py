import numpy as np
import argparse

from midap.utils import get_logger
from midap.data import DataProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--path_img", type=str, required=True,
                    help="Path to the original image used for the training. This image should be cut out already and "
                         "match the mask provided in the --path_mask argument.")
parser.add_argument("--path_mask", type=str, required=True,
                    help="Path to the mask (segmentation) used for the training. It should match the image provided to "
                         "the --path_mask argument.")
parser.add_argument("--path_train", type=str, required=True,
                    help="Name of the file used to save the training data.")
parser.add_argument("--path_test", type=str, required=True,
                    help="Name of the file used to save the testing data.")
parser.add_argument("--loglevel", type=int, default=7,
                    help="Loglevel of the script can range from 0 (no output) to 7 (debug, default)")
args = parser.parse_args()

# get the logger
logger =get_logger(__file__, logging_level=args.loglevel)

# Init the Processor
logger.debug("Initializing the DataProcessor...")
proc = DataProcessor(loglevel=args.loglevel)

# Create the data
path_img_train = args.path_img
path_mask_train = args.path_mask
logger.info(f"Using image data from: {path_img_train}")
logger.info(f"Using mask data from: {path_mask_train}")

# Preprocessing of data
(X_train,
y_train,
weight_maps_train,
ratio_cell_train,
X_val,
y_val,
weight_maps_val,
ratio_cell_val,
X_test,
y_test,
weight_maps_test,
ratio_cell_test) = proc.run(path_img_train,
                            path_mask_train)

# Save generated training data
np.savez_compressed(args.path_train, X_train = X_train,
y_train = y_train,
weight_maps_train = weight_maps_train,
ratio_cell_train = ratio_cell_train,
X_val = X_val,
y_val = y_val,
weight_maps_val = weight_maps_val,
ratio_cell_val = ratio_cell_val)

# Save generated test data
np.savez_compressed(args.path_test, X_test = X_test,
y_test = y_test,
weight_maps_test = weight_maps_test,
ratio_cell_test = ratio_cell_test)
