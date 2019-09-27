import numpy as np

import scipy.ndimage as ndi
from skimage.filters import sobel
from skimage.measure import label, regionprops
from skimage.morphology import watershed


def segment_image(img):
    # generate segmentation image
    elevation_map = sobel(img)

    markers = np.zeros_like(img)
    markers[img < 120] = 2
    markers[img > 40] = 1
    
    segm = watershed(elevation_map, markers)
    segm = ndi.binary_fill_holes(segm - 1)

    return segm

def remove_border_cell(segm):
    img_label = label(segm.astype(bool))

    for region in regionprops(img_label):
        minr, minc, maxr, maxc = region.bbox

        if 0 in region.bbox:
            segm[minr:maxr, minc:maxc] = 0
            
    return segm