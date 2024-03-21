import numpy as np
import matplotlib.pyplot as plt
import cv2
from plantcv import plantcv as pcv
from utils.rembg_ import rembg_


"""
Apply a mask on the image
"""

def mask_(image, col_space='GRAY'):
	# Create the mask using plantcv.threshold.binary. 
	# You will need to specify the input image, a binary threshold value, 
	# and the color space to use (e.g., 'LAB', 'GRAY', etc.).


	# Define the binary threshold value (adjust as needed)
	binary_threshold = 50

	# Convert the image to LAB color space (you can choose a different color space)
	if col_space == 'LAB':
		image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		# Create the binary mask
		binary_mask = pcv.threshold.binary(gray_img=image_lab[:, :, 0], threshold=binary_threshold)

	else:
		image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		binary_mask = pcv.threshold.binary(gray_img=image_lab, threshold=binary_threshold)


	# Apply the binary mask to the original image
	masked_image = pcv.apply_mask(img=image, mask=binary_mask, mask_color='black')

	return masked_image