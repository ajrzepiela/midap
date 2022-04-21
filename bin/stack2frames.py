from skimage import io
import argparse
import os
from tqdm import tqdm

from scipy.io import loadmat
from skimage.restoration import richardson_lucy

parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--pos")
parser.add_argument("--channel")
parser.add_argument("--start_frame")
parser.add_argument("--end_frame")
parser.add_argument("--deconv")
args = parser.parse_args()

path_head = os.path.split(args.path)[0]
path_tail = os.path.split(args.path)[1]
raw_filename = os.path.splitext(path_tail)[0]

# loop over tif/tiff-stack to extract single frames and deconvolve them if wanted
if args.deconv == "deconv_family_machine":
    psf = loadmat('../psf/PSFmme.mat')['PSF']
    deconvolution = True
elif args.deconv == "deconv_well":
    psf = io.imread('../psf/PSF_BornWolf.tif')[5,:,:]
    deconvolution = True
else:
    deconvolution = False
stack = io.imread(args.path)[int(args.start_frame):int(args.end_frame)]
for ix, frame in enumerate(tqdm(stack)):
        if deconvolution == True:
            deconvoluted = richardson_lucy(frame, psf, iterations=10, clip=False)
            io.imsave(path_head + '/' + 'raw_im/' + raw_filename + '_frame' + str("%03d" % (ix + int(args.start_frame))) + '_deconv.png', deconvoluted, check_contrast=False)
        else:
            io.imsave(path_head + '/' + 'raw_im/' + raw_filename + '_frame' + str("%03d" % (ix + int(args.start_frame))) + '.png', frame, check_contrast=False)

