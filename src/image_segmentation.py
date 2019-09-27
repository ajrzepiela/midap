import numpy as np

import scipy.ndimage as ndi
from skimage.filters import sobel
from skimage.morphology import watershed



def segment_image(img):
    elevation_map = sobel(img)

    markers = np.zeros_like(img)
    markers[img < 120] = 2
    markers[img > 40] = 1
    
    segm = watershed(elevation_map, markers)
    segm = ndi.binary_fill_holes(segm - 1)

    return segm