import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
from skimage.filters import sobel
from skimage.measure import label, regionprops
from skimage.morphology import watershed
from skimage.segmentation.boundaries import find_boundaries

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

def save_mat(cuts, segm, data_dir):
    # generate one mat-file per timeframe for segmentation images
    
    seg_dir = data_dir + 'xy1/seg/'
    fname_base = 'CWRt' 
    ix = 0
    for ph, seg in zip(cuts, segm):
    
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
                
        filename = seg_dir + fname_base + str('{:05}'.format(ix+1)) + 'xy001_seg'
        sio.savemat(filename,{'phase' : phase, 'mask_bg' : mask_bg.astype(float), 'mask_cell' : mask_cell.astype(bool), 
                            'crop_box' : crop_box,
                            'segs' : {'segs_good' : seg.astype(bool), 'segs_3n' : bounds.astype(bool),
                                                    'segs_label' : img_label.astype(bool), 'score' : score,
                                                    'scoreRaw' : scoreRaw, 'segs_bad' : seg.astype(bool)},
                            'regs' : {'regs_label' : img_label.astype(float), 'num_regs' : num_regs, 'score' : score,
                                                    'scoreRaw' : scoreRaw, 
                                        'props' :  properties}})

    ix+=1