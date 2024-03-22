from utils.rembg_ import rembg_
import random as rd
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Shear an image with a random factor

def shear_(path):
    # Load the image
    image = cv2.imread(path)

    # Remove background
    image = rembg_(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # # Define the shear transformation matrix
    shear_factor_x = rd.uniform(-1, 1)  # Shear factor along the x-axis
    # Calculate the new width of the sheared image
    new_width = int(image.shape[1] + abs(shear_factor_x) * image.shape[0])
    # new_height = int(image.shape[0] + abs(shear_factor_x) * image.shape[1])

    # Create a larger canvas
    sheared_image = np.zeros(
        (image.shape[0], new_width, image.shape[2]), dtype=np.uint8)

    # Calculate the translation to align the original image in the new canvas
    if shear_factor_x >= 0:
        translation_x = 0
    else:
        translation_x = int(abs(shear_factor_x) * image.shape[0])

    # Define the shear transformation matrix
    shear_matrix = np.array(
        [[1, shear_factor_x, translation_x], [0, 1, 0]], dtype=np.float32)

    # Apply the shear transformation
    sheared_image = cv2.warpAffine(
        image,
        shear_matrix,
        (new_width, image.shape[0]),
        borderValue=(0, 0, 0))
    sheared_image = cv2.resize(sheared_image, (image.shape[0], image.shape[1]))
    return sheared_image
