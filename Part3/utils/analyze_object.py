import numpy as np
import matplotlib.pyplot as plt
import cv2
from plantcv import plantcv as pcv
from utils.rembg_ import rembg_
from utils.mask_ import mask_


"""
Analyze object
"""

def analyze_object(image, points=[(80,80),(125,140)]):
	img = np.copy(image)

	# 
	mask = mask_(image, "GRAY")

	# 
	pcv.params.sample_label = "plant"

	# 
	thresh1 = pcv.threshold.dual_channels(rgb_img = image, x_channel = "a", y_channel = "b", points = points, above=True)

	# 
	a_fill_image = pcv.fill(bin_img=thresh1, size=50)
	a_fill_image = pcv.fill_holes(a_fill_image)

	# 
	roi1 = pcv.roi.rectangle(img=image, x=5, y=5, h=250, w=250)

	# 
	kept_mask  = pcv.roi.filter(mask=a_fill_image, roi=roi1, roi_type='partial')

	# 
	analysed_image = pcv.analyze.size(img=image, labeled_mask=kept_mask)
	return analysed_image