import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
from matplotlib.widgets  import RectangleSelector

from PIL import Image, ImageEnhance, ImageSequence
from scipy.ndimage.morphology import binary_dilation, grey_dilation
from skimage import util, exposure
#from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
from skimage.filters import threshold_otsu, threshold_mean
from skimage.filters import unsharp_mask
import skimage.measure as measure
import skimage.io as io
from skimage import img_as_float
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_filter
import cv2

# class LoadImage:

#     def __init__(self, path):
#         self.path = path

#     def open_tiff(self):
#         # load all pages of tiff file and return list of image arrays
#         im = Image.open(self.path)
#         all_pages = []
#         for i, page in enumerate(ImageSequence.Iterator(im)):
#             all_pages.append(np.array(page))
#         return all_pages


class CutoutImage:

    def __init__(self, files, min_x_range, max_x_range, min_y_range, max_y_range, ch1 = None, ch2 = None, cutout_mode = 'interactive'):
        #self.folder = folder
        self.files = files # list with files
        self.min_x_range = min_x_range
        self.max_x_range = max_x_range
        self.min_y_range = min_y_range
        self.max_y_range = max_y_range

        #additional channels
        self.ch1 = ch1 
        self.ch2 = ch2

        # mode for cutout
        self.cutout_mode = cutout_mode

        #adjust length of all channels       
        self.adjust_length()

######### needed methods
    def adjust_length(self):
        #set the lists with filenames of all channels to the same length
        length = [len(self.files)]

        if self.ch1 is not None:
            length.append(len(self.ch1))
        if self.ch2 is not None:
            length.append(len(self.ch2))

        min_len = np.min(length)

        self.files = self.files[:min_len]
        if self.ch1 is not None:
            self.ch1 = self.ch1[:min_len]
        if self.ch2 is not None:
            self.ch2 = self.ch2[:min_len]


    def align_two_images(self, src_img, ref_img):
        # aligns a source image in comparison to a reference image
        return phase_cross_correlation(src_img, ref_img)[0].astype(int)
        #return register_translation(src_img, ref_img)[0].astype(int)


    def align_all_images(self):
        # load 1st image of phase channel
        src = self.open_tiff(self.files[0])
        self.shifts = []
        for i in tqdm(range(1,len(self.files))):
            ref = self.open_tiff(self.files[i])
            # align image compared to 1st image
            shift = self.align_two_images(src, ref)
            self.shifts.append(shift)


    def set_offset(self):
	# computes the offset to choose depending on shift between images
        self.off = int(math.ceil(np.max(np.abs(self.shifts)) / 10.0)) * 10


    def cutout_single_frame(self, ix = None, img = None):
        # open tiff either from file or from array
        if ix is not None:
            if len(self.files) == 1:
                self.img = self.open_tiff(self.files[0])[ix]
            elif len(self.files) > 1:
                self.img = self.open_tiff(self.files[ix])
        elif img is not None:
            self.img = img

        if self.cutout_mode == 'automatic':
            # find contours in image
            contours = self.find_contour(self.img)
            # compute the range in x-direction for every contour and check for contours above a specific size
            ix_rect = self.find_rectangle(contours)
            # get the corners of the rectangle
            corners = self.get_corners(contours[ix_rect])
        
            # get coordinates from cutout and cutout images
            self.rectangle_x, self.rectangle_y, self.range_x, self.range_y = self.draw_rectangle(corners)
            self.corners_cut = self.get_corners(np.array([self.rectangle_y, self.rectangle_x]).T)

        elif self.cutout_mode == 'interactive':
            # interactive cutout of chambers
            corners = self.interactive_cutout()
            self.corners_cut = tuple([int(i) for i in corners])
       
        self.cutout = self.do_cutout(self.img, self.corners_cut)


    def open_tiff(self, path):
        # load all pages of tiff file and return list of image arrays
        im = io.imread(path)
        return im


    def find_contour(self, img):
        # find contour in image
 
        # binarize image
        thresh = threshold_mean(img)
        img_bin = np.zeros(img.shape)
        img_bin[img > thresh] = 1
        # define kernel
        kernel_vert = np.ones((50,1), np.uint8)
        kernel_hor = np.ones((1,50), np.uint8)
        # dilate image
        d_vert = cv2.dilate(img_bin, kernel_vert, iterations=1)
        d_hor = cv2.dilate(d_vert, kernel_hor, iterations=1)
        # find contours in binary image
        contours = measure.find_contours(d_hor, 0, fully_connected='high')
        return contours


    def find_rectangle(self, contours):
        # select only those contours which contain a closed shape
        closed_shape_ix = np.where([(c[:,1][0] == c[:,1][-1])&(c[:,0][0] == c[:,0][-1])\
                                                        for c in contours])[0]
        closed_shapes = [contours[ix] for ix in closed_shape_ix]

        #find the rectangular contour
        x_range_contours = np.array([np.max(c[:,1]) - np.min(c[:,1]) for c in closed_shapes])
        y_range_contours = np.array([np.max(c[:,0]) - np.min(c[:,0]) for c in closed_shapes])

        #assume chamber size is above a specific value
        ix = list(set(np.where(x_range_contours > self.min_x_range)[0])&\
        set(np.where(x_range_contours < self.max_x_range)[0])&\
        set(np.where(y_range_contours > self.min_y_range)[0])&\
        set(np.where(y_range_contours < self.max_y_range)[0]))
        ix_rect = closed_shape_ix[ix[0]]
        return ix_rect


    def get_corners(self, shape):
        # returns the corners of the rectangle
        left_x = int(np.min(shape[:,1]))
        right_x = int(np.max(shape[:,1]))
        lower_y = int(np.min(shape[:,0]))
        upper_y = int(np.max(shape[:,0]))
        return (left_x, right_x, lower_y, upper_y)


    def interactive_cutout(self):
        # generate interactive plot to select borders of chamber
        fig, self.ax = plt.subplots()
        self.ax.imshow(self.img)

        rs = RectangleSelector(self.ax, self.line_select_callback,
                       drawtype='box', useblit=False, button=[1],
                       minspanx=5, minspany=5, spancoords='pixels',
                       interactive=True)
        plt.show()

        left_x, right_x = rs.corners[0][:2]
        lower_y, upper_y = rs.corners[1][1:3]

        plt.imshow(self.img)
        plt.xlim([left_x, right_x])
        plt.ylim([lower_y, upper_y])
        plt.show()

        return (left_x, right_x, lower_y, upper_y)


    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
        self.ax.add_patch(rect)


    def draw_rectangle(self, corners, range_x = False, range_y = False, first_image = True):
        # draw a rectangle based on the corners
        left_x, right_x, lower_y, upper_y = corners
        if first_image:
            # draws a rectangle based on the corners
            left_x, right_x, lower_y, upper_y = corners
            range_x = right_x - left_x
            range_y = upper_y - lower_y

            rectangle_x = [left_x, right_x, right_x, left_x, left_x]
            rectangle_y = [lower_y, lower_y, upper_y, upper_y, lower_y]
            return rectangle_x, rectangle_y, range_x, range_y

        else:
            right_x = left_x + range_x
            lower_y = upper_y - range_y

            rectangle_x = [left_x, right_x, right_x, left_x, left_x]
            rectangle_y = [lower_y, lower_y, upper_y, upper_y, lower_y]
            return rectangle_x, rectangle_y


    def do_cutout(self, img, corners_cut):
        # generate cutout of image
        left_x, right_x, lower_y, upper_y = corners_cut
        cutout = img[lower_y:upper_y, left_x:right_x]
        return cutout


    def proc_image(self, img): 
        # postprocessing of image after cutout
        img_scaled = self.scale_pixel_val(img)
        return img_scaled


    def scale_pixel_val(self, img):
        # scale pixel values of image
        img_scaled = (255 * ((img - np.min(img))/np.max(img - np.min(img)))).astype('uint8')
        return img_scaled


    def shift_image(self, im, shift):
        # align images of additional channel
        return im[(self.off-shift[0]):(-self.off-shift[0]),
        (self.off-shift[1]):(-self.off-shift[1])]


    def save_cutout(self, files, file_names):
        # save of cutouts
        dir_name = os.path.dirname(os.path.dirname(file_names[0]))
        for f, i in zip(file_names, files):
            bname = os.path.basename(f).split('.')[0] + '_cut.png'
            io.imsave(dir_name + '/cut_im/' + bname, i, check_contrast = False)
########## end needed methods
        
    def run_align_cutout(self): # do all at once
        print('align images')
        self.align_all_images()
        self.set_offset()
        
        print('cutout images')
        src = self.open_tiff(self.files[0])
        src_off = src[self.off:-self.off, self.off:-self.off] #offset of 1st image
        
        # cutout of 1st image of all channel
        self.cutout_single_frame(img = src_off) #cutout of 1st image
        cut_src = self.proc_image(self.cutout)
        self.aligned_cutouts = [cut_src]
        if self.ch1 is not None: #1st additional channel
            src_ch1 = self.open_tiff(self.ch1[0])
            src_ch1_off = src_ch1[self.off:-self.off, self.off:-self.off] #offset of 1st image
            cut_src_ch1 = self.do_cutout(src_ch1_off, self.corners_cut)
            proc_src_ch1 = self.proc_image(cut_src_ch1)
            self.aligned_cutouts_ch1 = [proc_src_ch1] #cutout of 1st image
        if self.ch1 is not None: #2nd additional channel
            src_ch2 = self.open_tiff(self.ch2[0])
            src_ch2_off = src_ch2[self.off:-self.off, self.off:-self.off] #offset of 1st image	
            cut_src_ch2 = self.do_cutout(src_ch2_off, self.corners_cut)
            proc_src_ch2 = self.proc_image(cut_src_ch2)
            self.aligned_cutouts_ch2 = [proc_src_ch2] #cutout of 1st image	

        # cutout of all other images of all channels
        for i in tqdm(range(1,len(self.files))):
            ref = self.open_tiff(self.files[i])

            # align and cutout phase image compared to 1st image
            aligned_img = self.shift_image(ref, self.shifts[i-1])
            cut_img = self.do_cutout(aligned_img, self.corners_cut)
            proc_img = self.proc_image(cut_img)
            self.aligned_cutouts.append(proc_img)
            if self.ch1 is not None: #1st additional channel
                ref_ch1 = self.open_tiff(self.ch1[i])
                aligned_img_ch1 = self.shift_image(ref_ch1, self.shifts[i-1])
                cut_img_ch1 = self.do_cutout(aligned_img_ch1, self.corners_cut)
                proc_img_ch1 = self.proc_image(cut_img_ch1)
                self.aligned_cutouts_ch1.append(proc_img_ch1)
            if self.ch2 is not None: #2nd additional channel
                ref_ch2 = self.open_tiff(self.ch2[i]).astype(int)
                aligned_img_ch2 = self.shift_image(ref_ch2, self.shifts[i-1])
                cut_img_ch2 = self.do_cutout(aligned_img_ch2, self.corners_cut)
                proc_img_ch2 = self.proc_image(cut_img_ch2)
                self.aligned_cutouts_ch2.append(proc_img_ch2)
	
        # save cutouts
        self.save_cutout(self.aligned_cutouts, self.files)
        if self.ch1 is not None: #1st additional channel
            self.save_cutout(self.aligned_cutouts_ch1, self.ch1)
        if self.ch2 is not None: #2nd additional channel
            self.save_cutout(self.aligned_cutouts_ch2, self.ch2)

