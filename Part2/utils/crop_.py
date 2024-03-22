from utils.rembg_ import rembg_
import random as rd
import cv2
import warnings
warnings.filterwarnings('ignore')


# Crop
def crop_(path, shape_):
    image = cv2.imread(path)
    image = rembg_(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start_x = rd.randint(10, 50)
    end_x = rd.randint(200, 240)
    start_y = rd.randint(10, 50)
    end_y = rd.randint(200, 240)
    crop_image = image[start_x:end_x, start_y:end_y].copy(order='C')
    crop_image = cv2.resize(crop_image, (shape_[0], shape_[
                            1]), interpolation=cv2.INTER_AREA)
    return crop_image
