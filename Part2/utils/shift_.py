import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import AffineTransform, warp
import cv2
import random as rd
from utils.rembg_ import rembg_


def shift_(image, dx=25, dy=25):
	# X = x + dx
	# Y = y + dy
	# Here, dx and dy are the respective shifts along different dimensions.
	dx = rd.choice([-1, 1]) * rd.randint(15, 30)
	dy = rd.choice([-1, 1]) * rd.randint(15, 30)
	image = rembg_(image)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	tform = AffineTransform(translation=(dx,dy))
	shifted = warp(image, tform, cval=0)#, mode='wrap')
	return shifted