from utils.rembg_ import rembg_
import random as rd
import cv2
from skimage.filters import gaussian
import warnings
warnings.filterwarnings('ignore')

# Apply a Gaussian filter to an image


def gauss_(image):
    # Sigma here is the standard deviation for the Gaussian filter.
    # The higher the sigma value, the more will be the gaussian effect.
    sig = rd.uniform(1, 2)
    image = rembg_(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = rembg_(image)
    gauss = gaussian(image, sigma=sig)
    return gauss
