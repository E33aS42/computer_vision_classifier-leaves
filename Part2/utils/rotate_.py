import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import rotate
import cv2
import random as rd
from utils.rembg_ import rembg_
import rembg

# Rotate an image with a random angle
def rotate_(image, shape_):
	theta = rd.randint(10, 350)
	image = rembg_(image)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# image = rembg_(image)
	rotated = rotate(image, angle=theta, resize=True, cval=0)#, mode = 'wrap')
	# Setting mode as ‘wrap’ fills the points outside the boundaries of the input with the remaining pixels of the image.
	rotated = cv2.resize(rotated, (shape_[0], shape_[1]))
	return rotated