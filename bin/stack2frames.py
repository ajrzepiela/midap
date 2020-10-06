from skimage import io
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--pos")
parser.add_argument("--channel")
parser.add_argument("--start_frame")
parser.add_argument("--end_frame")
args = parser.parse_args()

path_head = os.path.split(args.path)[0]
path_tail = os.path.split(args.path)[1]
raw_filename = os.path.splitext(path_tail)[0]

stack = io.imread(args.path)[int(args.start_frame):int(args.end_frame)]
for ix, frame in enumerate(tqdm(stack)):
        io.imsave(path_head + '/' + 'raw_im/' + raw_filename + '_frame' + str("%03d" % ix) + '.tif', frame, check_contrast=False)

