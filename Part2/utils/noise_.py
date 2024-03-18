import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
import cv2
import random as rd
from utils.rembg_ import rembg_



# Adding random noise to an image

def add_noise(image):
	# standard deviation for noise to be added in the image
	# increasing this value will add more noise to the image and vice versa
	sig = rd.uniform(0.1, 0.5)
	image = rembg_(image)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# image = rembg_(image)
	noisyRandom = random_noise(image, var=sig**2)
	return noisyRandom
