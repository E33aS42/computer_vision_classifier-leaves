import rembg
import numpy as np
import warnings
warnings.filterwarnings('ignore')


""" remove background of an image """


def rembg_(image):
    input_array = np.array(image, order='C')
    # Apply background removal using rembg
    output_array = rembg.remove(input_array)
    # Create a PIL Image from the output array
    # output_image = Image.fromarray(output_array)
    return output_array
