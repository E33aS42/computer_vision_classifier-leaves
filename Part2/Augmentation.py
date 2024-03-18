# https://scikit-image.org/docs/dev/api/skimage.transform.html

from utils.distortion_ import distortion_
from utils.noise_ import add_noise
from utils.gauss_ import gauss_
from utils.shear_ import shear_
from utils.rotate_ import rotate_
from utils.shift_ import shift_
from utils.crop_ import crop_
from utils.flip_ import flip_
from utils.rembg_ import rembg_
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import glob
import warnings
warnings.filterwarnings('ignore')


# remove background of the original image
def original_(image):
    image = rembg_(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# apply and save images augmentations

def get_img_name(path):
    if '/' in path:
        filename = path.rsplit('/', 1)[1]
        name = filename.split('.')[0]
    else:
        name = path.split('.')[0]
    return name


def get_dir_name(path):
    path_split = path.rsplit('/')
    if path_split[-1] == "":
        return path.rsplit('/')[-2]
    else:
        return path.rsplit('/')[-1]


def plot_img_augm(fig, img, name, augm, nb):
    columns = 8
    fig.add_subplot(1, columns, nb)
    plt.axis('off')
    plt.title(augm)
    plt.imshow(img)
    if augm != "Original":
        save_img_augm(img, "", name, augm)


def save_img_augm(img, name_dir, name, augm):
    if augm == "Original":
        name_dir = name_dir + "/" + augm
    if not os.path.isdir("../augmented_directory"):
        os.makedirs("../augmented_directory")
    if name_dir == "":
        filename = "../augmented_directory/" + name + "_" + augm + ".JPG"
    else:
        if not os.path.isdir("../augmented_directory" + "/" + name_dir):
            os.makedirs("../augmented_directory" + "/" + name_dir)
        filename = "../augmented_directory/" + \
            name_dir + "/" + name + "_" + augm + ".JPG"
    plt.imsave(filename, img)


def image_augm(path, name):
    image = cv2.imread(path)
    # image = Image.open(path)

    # shape of the image
    shape_ = image.shape

    # Plot different images augmentations
    fig = plt.figure()
    plot_img_augm(fig, original_(image), name, "Original", 1)
    plot_img_augm(fig, flip_(image), name, "Flip", 2)
    plot_img_augm(fig, crop_(path, shape_), name, "Crop", 3)
    plot_img_augm(fig, rotate_(image, shape_), name, "Rotate", 4)
    plot_img_augm(fig, shear_(path), name, "Shear", 5)
    plot_img_augm(fig, shift_(image), name, "Shift", 6)
    plot_img_augm(fig, add_noise(image), name, "Noise", 7)
    plot_img_augm(fig, distortion_(path, shape_), name, "Distortion", 8)
    plt.show()


def plot_img_augm_folder(fig, img, name_dir, name, augm, i, nb):
    rows = 6
    columns = 7
    fig.add_subplot(rows, columns, nb + columns * (i - 1))
    plt.axis('off')
    if i == 1:
        plt.title(augm)
    plt.imshow(img)
    if augm != "Original":
        save_img_augm(img, name_dir, name, augm)


def image_augm_folder(fig, path, name_dir, name, i):
    image = cv2.imread(path)

    # shape of the image
    shape_ = image.shape

    if i < 7:
        # Plot and save different images augmentations
        plot_img_augm_folder(fig, original_(
            image), name_dir, name, "Original", i, 1)
        plot_img_augm_folder(fig, flip_(image), name_dir, name, "Flip", i, 2)
        plot_img_augm_folder(fig, crop_(path, shape_),
                             name_dir, name, "Crop", i, 3)
        plot_img_augm_folder(fig, rotate_(image, shape_),
                             name_dir, name, "Rotate", i, 4)
        plot_img_augm_folder(fig, shear_(path), name_dir, name, "Shear", i, 5)
        plot_img_augm_folder(fig, shift_(image), name_dir,
                             name, "Shift", i, 6)
        plot_img_augm_folder(fig, add_noise(
            image), name_dir, name, "Noise", i, 7)

    else:
        # Save improved original image and different images augmentations
        save_img_augm(original_(image), name_dir, name, "Original")
        save_img_augm(flip_(image), name_dir, name, "Flip")
        save_img_augm(crop_(path, shape_), name_dir, name, "Crop")
        save_img_augm(rotate_(image, shape_), name_dir, name, "Rotate")
        save_img_augm(shear_(path), name_dir, name, "Shear")
        save_img_augm(shift_(image), name_dir, name, "Shift")
        save_img_augm(add_noise(image), name_dir, name, "Noise")


if __name__ == "__main__":
    try:

        sys_argv_length = len(sys.argv)
        if len(sys.argv) >= 2:

            path = sys.argv[1]

            # Checks if path is a file
            if os.path.isfile(path):
                print("isFile")
                name = get_img_name(path)
                image_augm(path, name)

            # Checks if path is a directory
            elif os.path.isdir(path):
                print("isDir")
                name_dir = get_dir_name(path)
                i = 1
                fig = plt.figure()
                for imgpath in glob.iglob(f'{path}/*'):
                    if os.path.isfile(imgpath):
                        name = get_img_name(imgpath)
                        image_augm_folder(fig, imgpath, name_dir, name, i)
                        i += 1
                    else:
                        print("isnotFile")
                plt.show()

        else:
            print("Missing argument. Try again.")
            exit()

    except Exception as e:
        print(e)
