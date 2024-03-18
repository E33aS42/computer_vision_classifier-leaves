import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import cv2
import random as rd
from utils.rembg_ import rembg_
import rembg

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
	