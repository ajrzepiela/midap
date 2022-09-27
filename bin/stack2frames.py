import argparse

import os
from tqdm import tqdm

from scipy.io import loadmat
from skimage.restoration import richardson_lucy
from skimage import io

from utils import get_logger

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="Path to the file of which the frames should be split")
parser.add_argument("--start_frame", type=int, required=True, help="First frame to split off.")
parser.add_argument("--end_frame", type=int, required=True, help="Last frame to split off.")
parser.add_argument("--deconv", type=str, choices=["deconv_family_machine", "deconv_well", "no_deconv"],
                    default="no_deconv", help="Deconvolution type that should be performed, defaults to no deconv.")
parser.add_argument("--loglevel", type=int, default=7, help="Loglevel of the script.")
args = parser.parse_args()

# logging
logger = get_logger(__file__, args.loglevel)
logger.info(f"Splitting frames of: {args.path}")

# split the paths
path_head, path_tail = os.path.split(args.path)
raw_filename = os.path.splitext(path_tail)[0]

# loop over tif/tiff-stack to extract single frames and deconvolve them if wanted
if args.deconv == "deconv_family_machine":
    logger.debug("Running deconv for family machine.")
    psf = loadmat(os.path.join('..', 'psf', 'PSFmme.mat'))['PSF']
    deconvolution = True
elif args.deconv == "deconv_well":
    logger.debug("Running deconv for well.")
    psf = io.imread(os.path.join('..', 'psf', 'PSF_BornWolf.tif'))[5,:,:]
    deconvolution = True
else:
    logger.debug("No deconv selected")
    deconvolution = False

# split the frames
logger.info("Splitting frames...")
stack = io.imread(args.path)[int(args.start_frame):int(args.end_frame)]
for ix, frame in enumerate(tqdm(stack)):
        if deconvolution:
            deconvoluted = richardson_lucy(frame, psf, iterations=10, clip=False)
            io.imsave(os.path.join(path_head, 'raw_im', f'{raw_filename}_frame{ix + args.start_frame:03d}_deconv.png'),
                      deconvoluted, check_contrast=False)
        else:
            io.imsave(os.path.join(path_head, 'raw_im', f'{raw_filename}_frame{ix + args.start_frame:03d}.png'),
                      frame, check_contrast=False)

