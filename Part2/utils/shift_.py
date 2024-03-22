from utils.rembg_ import rembg_
import random as rd
import cv2
from skimage.transform import AffineTransform, warp
import warnings
warnings.filterwarnings('ignore')


def shift_(image, dx=25, dy=25):
    # X = x + dx
    # Y = y + dy
    # Here, dx and dy are the respective shifts along different dimensions.
    dx = rd.choice([-1, 1]) * rd.randint(15, 30)
    dy = rd.choice([-1, 1]) * rd.randint(15, 30)
    image = rembg_(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tform = AffineTransform(translation=(dx, dy))
    shifted = warp(image, tform, cval=0)  # , mode='wrap')
    return shifted
