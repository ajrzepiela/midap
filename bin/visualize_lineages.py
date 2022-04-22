import argparse
import napari
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='path to folder with tracking output')
args = parser.parse_args()

data = np.load(args.path + 'label_stack.npz')
label_stack = data['label_stack']

with open(args.path + 'tracks_data.pkl', 'rb') as f:
    tracks_data = pickle.load(f)

viewer = napari.view_image(label_stack, name='image')
viewer.add_tracks(tracks_data, name='tracks')


napari.run()