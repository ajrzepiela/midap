import cv2
from PIL import Image, ImageEnhance
from skimage.filters import unsharp_mask
from skimage import io

from multiprocessing import Pool
import numpy as np

class Preprocessing:

    def __init__(self, n_cores):
        self.n_cores = n_cores

    def full_preprocessing(self, path = None, img = None):
        if path:
            img = self.load_img(path)
        elif img is not None and isinstance(img, (list, tuple, np.ndarray)):
            img = Image.fromarray(img)
        img_enhanced = self.increase_contrast(img)
        imgs_sharp, sharpness = self.fit_masking(img_enhanced)
        img_sharp = self.find_best_fit(sharpness, imgs_sharp)
        return img_sharp

    def load_img(self, path):
        return Image.open(path)

#     def invert_img(self, img):
#         if np.mean(img) > int(img.max()/2):
#             return np.invert(img)
#         else:
#             return img

    def increase_contrast(self, img):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(2.0)

    def sharpen_img(self, img, radius, amount):
        img_sharp = unsharp_mask(img, radius, amount)
        return self.normalize_pixels(img_sharp)

    def normalize_pixels(self, img):
        return ((img - img.min())/(img.max()) * 255).astype('uint8')

    def compute_sharpness(self, img):
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def test_vals(self, img, radius, amount):
        img_sharp = self.sharpen_img(img, radius, amount)
        sharpness = self.compute_sharpness(img_sharp)
        return img_sharp, sharpness

    def find_best_fit(self, sharpness, imgs_sharp):
        ix_sharp = np.argmax(sharpness)
        return imgs_sharp[ix_sharp]

    def fit_masking(self, img):
        radius_array = np.arange(1,11,1)
        amount_array = np.arange(1,11,1)

        radius_mesh, amount_mesh = np.meshgrid(radius_array, amount_array)

        #result = [self.test_vals(img, r, a) for r, a in zip(radius_mesh.flatten(), amount_mesh.flatten())]
        num_repeats = len(radius_mesh.flatten())
        imgs = [img]*num_repeats
        p = Pool(self.n_cores)
        result = p.starmap(self.test_vals, zip(imgs, radius_mesh.flatten(), amount_mesh.flatten()))
        imgs_sharp = [r[0] for r in result]
        sharpness = [r[1] for r in result]

        return imgs_sharp, sharpness
    
