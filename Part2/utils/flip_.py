from utils.rembg_ import rembg_
import random as rd
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')


""" randomly flip an image vertically or horizontally

"""


def flip_(image):
    image = rembg_(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    i = rd.randint(0, 1)
    if i == 0:
        # flip image vertically
        flip_img = np.flipud(image)
    else:
        # flip image horizontally
        flip_img = np.fliplr(image)

    # flip_img = rembg_(flip_img)
    return flip_img
