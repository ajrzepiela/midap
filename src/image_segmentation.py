import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
from skimage import feature
from skimage import exposure
from skimage.filters import sobel, threshold_otsu, threshold_minimum, threshold_local
from skimage.measure import label, regionprops
from skimage.morphology import watershed
from skimage.segmentation.boundaries import find_boundaries
from skimage.util import img_as_ubyte
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage.exposure import is_low_contrast, adjust_sigmoid
from skimage.feature import canny
from skimage.filters import unsharp_mask
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.filters import rank
from skimage.morphology import disk
from scipy.ndimage.morphology import binary_fill_holes

from image_preprocessing import Preprocessing

import matplotlib.pyplot as plt

class ImageSegmentation:

    def __init__(self, channel, n_cores, mode = None, min_pixel_val = None, max_pixel_val = None):
        self.channel = channel
        self.n_cores = n_cores
        self.mode = mode
        if min_pixel_val:
            self.min_pixel_val= min_pixel_val
        if max_pixel_val:
            self.max_pixel_val = max_pixel_val

    # def __init__(self, all_imgs, min_pixel_val, max_pixel_val, mode = 'watershed'):
    #     self.all_imgs = all_imgs
    #     self.min_pixel_val= min_pixel_val
    #     self.max_pixel_val = max_pixel_val
    #     self.mode = mode

    def invert_img(self, img):
        if np.mean(img) > int(255/2):
            return np.invert(img)
        else:
            return img

    def preprocess_img(self, img):
        p = Preprocessing(self.n_cores)
        return p.full_preprocessing(img = img)


    def segment_thresh_otsu(self, img):
        #img = self.invert_img(img) 
        otsu = threshold_otsu(img)
        #otsu = np.mean(img)
        segmentation = img > otsu
        return segmentation


    def segment_single_frame_edge(self, img):
        edges = feature.canny(np.array(img))
        segmentation = ndi.morphology.binary_fill_holes(edges)
         
        seg_cleaned = self.clean_segmentation(segmentation)
        label_img = label(seg_cleaned)
        #label_objects, _ = ndi.label(fill_cells)
        #sizes = np.bincount(label_objects.ravel())
        #mask_sizes = sizes > 120
        #mask_sizes[0] = 0
        #segmentation = mask_sizes[label_objects]
        return label_img    


    def segment_region_based(self, img, min_val, max_val):
        elevation_map = sobel(img)
        markers = np.zeros_like(img)
        markers[img < min_val] = 1
        markers[img > max_val] = 2
        segmentation = watershed(elevation_map, markers)
        segmentation = binary_fill_holes(segmentation - 1)
        return segmentation


    def segment_kmeans_region_based(self, img):
        res = self.kmeans_clustering(img)

        range_background = np.unique(img[res == np.unique(res)[0]])
        range_cell = np.unique(img[res == np.unique(res)[-1]])

        segmentation = self.segment_region_based(img,range_background.max(), range_cell.min())
        return segmentation
    
    def segment_enhanced_contrast(self, img, disk_param):
        #img = np.array(img).astype('uint8')
        img_auto = rank.enhance_contrast(img, disk(disk_param))
        pixel_vals = np.unique(img_auto)
        print(pixel_vals)
        if len(pixel_vals) > 1:
            ix_thresh = np.where(np.diff(pixel_vals) > 2)[0][0]
            pixel_thresh = pixel_vals[ix_thresh + 1]
            segmentation = (img_auto > pixel_thresh).astype(int)
        else:
            segmentation = img_auto
        return segmentation


    def remove_border_cell(self, img):
        img_label = label(img.astype(bool))

        for region in regionprops(img_label):
            minr, minc, maxr, maxc = region.bbox

            if 0 in region.bbox:
                img_label[minr:maxr, minc:maxc] = 0
        return img_label

    def recreate_image(self, codebook, labels, w, h):
        image = np.zeros((w, h))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    def kmeans_clustering(self, img):
        n_colors = 16
        img = np.array(img, dtype=np.float64) / img.max()

        w, h = tuple(img.shape)
        image_array = np.reshape(img, (w * h, 1))

        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)

        return self.recreate_image(kmeans.cluster_centers_, labels, w, h)

    def remove_cells(self, seg):
        seg_label = label(seg)
        seg_clean = np.copy(seg_label)

        component_sizes = np.bincount(seg_label.ravel())
        too_small = component_sizes < 100
        too_small_mask = too_small[seg_label]
        seg_clean[too_small_mask] = 0

        too_big = component_sizes > 400
        too_big_mask = too_big[seg_label]
        seg_clean[too_big_mask] = 0
        #print(np.unique(seg_clean))
        seg_clean[seg_clean > 0] = 1
        #print(np.unique(seg_clean))
        return seg_clean
        

    def clean_segmentation(self, seg):
        seg_cleaned = self.remove_cells(seg)
        seg_cleaned = self.remove_border_cell(seg_cleaned)
        return seg_cleaned


    #def segment_single_frame_phase(self, img):
        # generate segmentation image
        # elevation_map = sobel(img)

        # markers = np.zeros_like(img)
        # markers[img < self.min_pixel_val] = 1
        # markers[img > self.max_pixel_val] = 2

        # wat = watershed(elevation_map, markers)
        # segmentation = ndi.binary_fill_holes(wat) #-1
        # return segmentation


    def segment_single_frame_fluor(self, img):
        #img = self.invert_img(img) 
        img_sharp = self.preprocess_img(img) 
        img_cluster = self.kmeans_clustering(img_sharp)

        edges_cluster = canny(img_cluster)
        edges_sharp = canny(img_sharp)

        combined = edges_cluster + edges_sharp
        segmentation = binary_fill_holes(combined)
        return segmentation


    def segment_single_frame(self, img):
        if self.mode == 'region_based':
            if self.channel == 'eGFP' or self.channel == 'mCherry':
                segmentation = self.segment_region_based(img, min_val = 30, max_val = 70)
            elif self.channel == 'phase':
                segmentation = self.segment_region_based(img, min_val = 110, max_val = 190)
        elif self.mode == 'edge':
                segmentation = self.segment_single_frame_edge(img)
        elif self.mode == 'k_means':
                segmentation = self.segment_single_frame_fluor(img)
        elif self.mode == 'k_means_region_based':
            if self.channel == 'phase' or self.channel == 'eGFP':
                img_adjust = adjust_sigmoid(img)
                segmentation = self.segment_kmeans_region_based(img_adjust)
            elif self.channel == 'mCherry':
                segmentation = self.segment_kmeans_region_based(img)
        elif self.mode == 'enhance_contrast':
            if self.channel == 'phase':
                segmentation = self.segment_region_based(np.invert(img), min_val = 120, max_val = 180)
            elif self.channel == 'mCherry':
                segmentation = self.segment_enhanced_contrast(img, 10)
            elif self.channel == 'eGFP':
                segmentation = self.segment_enhanced_contrast(img, 20)
        seg_cleaned = self.clean_segmentation(segmentation)
        label_img = label(seg_cleaned)

        return label_img


    # def segment_single_frame(self, ix):
    #     self.img = self.all_imgs[ix]
    #     if self.mode == 'watershed':
    #         self.segment_image(self.img)
    #     elif self.mode == 'otsu':
    #         self.segment_thresh_otsu(self.img)
    #     elif self.mode == 'edge':
    #         self.segment_edge(self.img)
    #     self.remove_border_cell()

    def count_cells(self, img):
        return np.max(img)

    
    def segment_all_frames(self, imgs):
        self.all_segms = []
        self.all_nums = []
        for i in tqdm(imgs):
            label_img = self.segment_single_frame(i)
            num_cells = self.count_cells(label_img)
            self.all_segms.append(label_img)
            self.all_nums.append(num_cells)


def save_mat(cuts, segm, data_dir):
    # generate one mat-file per timeframe for segmentation images

    seg_dir = data_dir + 'xy1/seg/'
    fname_base = 'CWRt' 
    ix = 0
    for ph, seg in zip(cuts, segm):
        print('run')
        #phase images
        phase = ph

        # mask background
        mask_bg = np.where(((seg * 1)<1),0,1).astype('uint8')

        # labeling of segmentations
        img_label = label(seg.astype(bool))
        bounds = find_boundaries(img_label)
        score = np.ones(np.max(img_label)).astype(int)
        scoreRaw = score * 50

        # detect and count all cells in image
        num_regs = np.max(img_label)#int(len(np.unique(img_label)))

        props = regionprops(img_label)

        properties = []
        for p in props:

            min_col = p.bbox[1]
            min_row = p.bbox[0]
            diff_col = p.bbox[3] - p.bbox[1]
            diff_row = p.bbox[2] - p.bbox[0]
            bb = tuple([min_col+1, min_row+1, diff_col+1, diff_row+1])

            bb_dict = np.zeros(1, dtype = [('Area', int, (1, 1)), ('BoundingBox', float, (1, 4))])
            bb_dict['Area'] = p.area
            bb_dict['BoundingBox'] = bb

            properties.append(bb_dict)

        # vars to be defined
        mask_cell = (mask_bg - bounds - seg) > 0

        # image size and values
        min_col_bb = np.min([p[0][1][0][0] for p in properties])
        min_row_bb = np.min([p[0][1][0][1] for p in properties])

        max_col_bb = np.max([p[0][1][0][0] for p in properties])
        max_row_bb = np.max([p[0][1][0][1] for p in properties])

        crop_box = np.array([[min_row_bb, min_col_bb, max_row_bb, max_col_bb]]).astype(int)

        filename = seg_dir + fname_base + str('{:05}'.format(ix+1)) + 'xy001_seg.mat'
        sio.savemat(filename,{'phase' : phase, 'mask_bg' : mask_bg.astype(float), 'mask_cell' : mask_cell.astype(bool), 
                            'crop_box' : crop_box,
                            'segs' : {'segs_good' : seg.astype(bool), 'segs_3n' : bounds.astype(bool),
                                                    'segs_label' : img_label.astype(bool), 'score' : score,
                                                    'scoreRaw' : scoreRaw, 'segs_bad' : seg.astype(bool)},
                            'regs' : {'regs_label' : img_label.astype(float), 'num_regs' : num_regs, 'score' : score,
                                                    'scoreRaw' : scoreRaw, 
                                        'props' :  properties}})


        ix+=1
