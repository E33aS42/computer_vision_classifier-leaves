import cv2
from plantcv import plantcv as pcv
from utils.mask_ import mask_


"""
Apply a black & white Gaussian blur on the image
"""


def Gblur_(image, kernel="(21, 21)"):
    # img - RGB or grayscale image data
    # ksize - Tuple of kernel dimensions, e.g. (5, 5). Must be odd integers.
    # sigma_x - standard deviation in X direction; if 0 (default), calculated from kernel size
    # sigma_y - standard deviation in Y direction; if sigma_Y is None (default), sigma_Y is taken to equal sigma_X

    # apply a mask if wanted, otherwise comment this line
    image = mask_(image, "LAB")

    # switched image to Black & White
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gaussian_img = pcv.gaussian_blur(
        img=image, ksize=(21, 21), sigma_x=0, sigma_y=None)

    return gaussian_img
