import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from PIL import Image, ImageTransform
import cv2
import random as rd
from utils.rembg_ import rembg_
import rembg


def distortion_(path, shape_):
	# Load the image
	image = cv2.imread(path)
	image = rembg_(image)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# image = rembg_(image)
	
	# Define the distortion factors
	distortion_factor_x = 0.4  # Horizontal distortion factor
	distortion_factor_y = 1.0  # Vertical distortion factor (no distortion)

	# Create the distortion transformation matrix
	distortion_matrix = np.array([[1.0 + distortion_factor_x, 0, 0],
								[0, 1.0, 0]], dtype=np.float32)

	# Apply the distortion transformation
	distorted_image = cv2.warpAffine(image, distortion_matrix, (image.shape[1], image.shape[0]))
	return distorted_image