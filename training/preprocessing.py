import os.path

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
parser.add_argument("--n_grid", type=int, default=4,
                    help="The grid used to split the original image into distinct patches for train, test and val "
                         "dsets")
parser.add_argument("--test_size", type=float, default=0.15,
                    help="Ratio for the test set")
parser.add_argument("--val_size", type=float, default=0.2,
                    help="Ratio for the validation set")
parser.add_argument("--patch_size", type=int, default=128,
                    help="The size of the patches of the final square images (not relevant for mother machine)")
parser.add_argument("--num_split_r", type=int, default=10,
                    help="number of patches along row dimension for the patch generation "
                         "(not relevant for mother machine)")
parser.add_argument("--num_split_c", type=int, default=10,
                    help="number of patches along column dimension for the patch generation "
                         "(not relevant for mother machine)")
parser.add_argument("--no_augmentation", action="store_false",
                    help="Perform no data augmentation of the patches")
parser.add_argument("--sigma", type=float, default=2.0,
                    help="sigma parameter used for the weight map calculation (not relevant for mother machine)")
parser.add_argument("--w_0", type=float, default=2.0,
                    help="w_0 parameter used for the weight map calculation (not relevant for mother machine)")
parser.add_argument("--w_c0", type=float, default=1.0,
                    help="basic class weight for non-cell pixel parameter used for the weight map calculation "
                         "(not relevant for mother machine)")
parser.add_argument("--w_c1", type=float, default=1.1,
                    help="basic class weight for cell pixel parameter used for the weight map calculation "
                         "(not relevant for mother machine)")
parser.add_argument("--np_random_seed", type=int, default=None,
                    help="A random seed for the numpy random seed generator, defaults to None, which will lead "
                         "to non-reproducible behaviour.")
parser.add_argument("--mother_machine", action="store_true",
                    help="Flag to indicate a mother machine run.")
parser.add_argument("--loglevel", type=int, default=7,
                    help="Loglevel of the script can range from 0 (no output) to 7 (debug, default)")
args = parser.parse_args()

# get the logger
logger =get_logger(__file__, logging_level=args.loglevel)

# check if we have a random state
if args.np_random_seed is None:
    logger.warning("No random state was set, this means the output will not be reproducible.")

# Init the Processor
logger.debug("Initializing the DataProcessor...")
proc = DataProcessor(n_grid=args.n_grid, test_size=args.test_size, val_size=args.val_size,
                     patch_size=args.patch_size, num_split_r=args.num_split_r, num_split_c=args.num_split_c,
                     augment_patches=~args.no_augmentation, sigma=args.sigma, w_0=args.w_0, w_c0=args.w_c0,
                     w_c1=args.w_c1, loglevel=args.loglevel, np_random_seed=args.np_random_seed)

# Get the data
path_img_train = args.path_img
path_mask_train = args.path_mask
logger.info(f"Using image data from: {path_img_train}")
logger.info(f"Using mask data from: {path_mask_train}")

# Create the save directories if necessary
save_dir, _ = os.path.split(os.path.abspath(args.path_train))
os.makedirs(save_dir, exist_ok=True)
save_dir, _ = os.path.split(os.path.abspath(args.path_test))
os.makedirs(save_dir, exist_ok=True)

# Preprocessing of data
if args.mother_machine:
    X_train, y_train, X_val, y_val = proc.run_mother_machine(path_img_train, path_mask_train)

    # Save generated training data
    np.savez_compressed(args.path_train, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
else:
    data = proc.run(path_img_train, path_mask_train)

    # Save generated training data
    np.savez_compressed(args.path_train, X_train=data["X_train"], y_train=data["y_train"],
                        weight_maps_train=data["weight_maps_train"], ratio_cell_train=data["ratio_cell_train"],
                        X_val=data["X_val"], y_val=data["y_val"], weight_maps_val=data["weight_maps_val"],
                        ratio_cell_val=data["ratio_cell_val"])

    # Save generated test data
    np.savez_compressed(args.path_test, X_test=data["X_test"], y_test=data["y_test"],
                        weight_maps_test=data["weight_maps_test"], ratio_cell_test=data["ratio_cell_test"])
