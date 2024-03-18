import numpy as np
import matplotlib.pyplot as plt
import cv2
from plantcv import plantcv as pcv
from utils.rembg_ import rembg_


"""
Keypoints detection
"""

def keypoints(image, nb=30):

	img = np.copy(image)

	# Initialize ORB detector giving number nb of keypoints to find
	orb = cv2.ORB_create(nb)

	# Detect key points and compute descriptors
	keypoints, descriptors = orb.detectAndCompute(img, None)
	img_kp = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	return img_kp