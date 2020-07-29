from skimage import io
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--pos")
parser.add_argument("--channel")
args = parser.parse_args()

path_head = os.path.split(args.path)[0]
path_tail = os.path.split(args.path)[1]
raw_filename = os.path.splitext(path_tail)[0]

stack = io.imread(args.path)
print(stack.shape)
for ix, frame in enumerate(tqdm(stack[:10])):
	io.imsave(path_head + '/' + args.pos + args.channel + 'raw_im/' + raw_filename + '_frame' + str("%03d" % ix) + '.tif', frame, check_contrast=False)

