from utils.rembg_ import rembg_
# import random as rd
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def distortion_(path, shape_):
    # Load the image
    image = cv2.imread(path)
    image = rembg_(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = rembg_(image)

    # Define the distortion factors
    distortion_factor_x = 0.4  # Horizontal distortion factor
    # distortion_factor_y = 1.0  # Vertical distortion factor (no distortion)

    # Create the distortion transformation matrix
    distortion_matrix = np.array([[1.0 + distortion_factor_x, 0, 0],
                                  [0, 1.0, 0]], dtype=np.float32)

    # Apply the distortion transformation
    distorted_image = cv2.warpAffine(
        image, distortion_matrix, (image.shape[1], image.shape[0]))
    return distorted_image
