from skimage import io
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm

img = io.imread('../data/data_Johannes/unet/ImprovedStackWell_frame33_OrigIm_cutout.tif')
mask = io.imread('../data/data_Johannes/unet/ImprovedStackWell_frame33_MaskOnly_Edt_cutout.tif')

label_mask = label(mask)
label_props = regionprops(label_mask)

splitting_events = []
new_im = np.zeros(mask.shape)

for lp in tqdm(label_props):
    minr, minc, maxr, maxc = lp.bbox

    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(2,1,1)
    ax1.imshow(img)

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect)
    plt.ylim(np.max([0,minr-10]), np.min([mask.shape[0], maxr + 10]))
    plt.xlim(np.max([0,minc-10]), np.min([mask.shape[1], maxc + 10]))

    ax2 = plt.subplot(2,1,2)
    ax2.imshow(mask)

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
    ax2.add_patch(rect)
    plt.ylim(np.max([0,minr-10]), np.min([mask.shape[0], maxr + 10]))
    plt.xlim(np.max([0,minc-10]), np.min([mask.shape[1], maxc + 10]))

    plt.show()

    inp = input("Is this a splitting event? ")
    splitting_events.append(inp)

    if inp == 'y':
        for c in lp.coords:
            new_im[c[0],c[1]] = 1

    if inp == 'n':
        for c in lp.coords:
            new_im[c[0],c[1]] = 2

np.save('label_splitting_cells.npy', splitting_events)
np.save('mask_splitting_cells.npy', new_im)