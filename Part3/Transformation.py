#!/usr/bin/env python

from utils.color_hist import color_hist, color_hist_save
from utils.keypoints import keypoints
from utils.pseudolandmarks import Pseudolandmarks, Pseudolandmarks_fig
from utils.analyze_object import analyze_object
from utils.mask_ import mask_
from utils.gblur_ import Gblur_
from utils.rembg_ import rembg_
from utils.helperfile import helper
import cv2
import matplotlib.pyplot as plt
import sys
import os
import glob
import warnings
warnings.filterwarnings('ignore')


"""
PlantCV is an open-source image analysis software package targeted
for plant phenotyping.
Phenotyping means the quantitative analysis of plant structures and functions.
A phenotype is the set of observable characteristics or traits of an organism.

"""


def get_img_name(path):
    if '/' in path:
        filename = path.rsplit('/', 1)[1]
    else:
        filename = path
    name = filename.split('.')[0]
    return name


def get_dir_name(path):
    return path.rsplit('/', 1)[1]


def image_augm(path, name):
    img = cv2.imread(path)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = rembg_(img)

    # Plot different images augmentations
    fig = plt.figure()
    columns = 3
    lines = 2
    fig.tight_layout(pad=25.0)

    fig.add_subplot(lines, columns, 1)
    plt.imshow(img)
    plt.xlabel('Figure IV.1: Original')

    fig.add_subplot(lines, columns, 2)
    plt.imshow(Gblur_(img), cmap='gray')
    plt.xlabel('Figure IV.2: Gaussian blur')

    fig.add_subplot(lines, columns, 3)
    # plt.imshow(mask_(img, "LAB"))
    plt.imshow(mask_(img, "GRAY"))
    plt.xlabel('Figure IV.3: Mask')

    fig.add_subplot(lines, columns, 4)
    plt.imshow(analyze_object(img), cmap='gray')
    plt.xlabel('Figure IV.4: Analyze object')

    ax = fig.add_subplot(lines, columns, 5)
    Pseudolandmarks(img, ax)
    plt.xlabel('Figure IV.5: Pseudolandmarks')

    fig.add_subplot(lines, columns, 6)
    plt.imshow(keypoints(img, 30))
    plt.xlabel('Figure IV.6: Keypoints detection')

    color_hist(img)

    plt.show()


def save_image_transf_folder(dest, imgpath, name, type_):
    """
    apply type_ tranformation to images folder
    and save them into a type_ folder inside the dest folder
    """
    image = cv2.imread(imgpath)
    if not os.path.isdir(dest):
        os.makedirs(dest)
    type_ = type_[1:]
    dest_folder = dest + "/" + type_
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    function_dict = {
        "mask": mask_,
        "Gblur": Gblur_,
        "landmarks": Pseudolandmarks_fig,
        "analyze": analyze_object,
        "keypoints": keypoints,
        "histo": color_hist_save}

    augm = function_dict[type_](image)
    filename = dest_folder + "/" + name + "_" + type_ + ".JPG"
    # print(filename)

    cv2.imwrite(filename, augm)


def save_image_all_transf(dest, imgpath, name):
    """
    apply all tranformations to images folder and save them into dest folder
    """

    image = cv2.imread(imgpath)
    if not os.path.isdir(dest):
        os.makedirs(dest)
    function_dict = {
        "mask": mask_,
        "Gblur": Gblur_,
        "landmarks": Pseudolandmarks_fig,
        "analyze": analyze_object,
        "keypoints": keypoints,
        "histo": color_hist_save}

    for key in function_dict.keys():
        augm = function_dict[key](image)
        dest_folder = dest + "/" + key
        if not os.path.isdir(dest_folder):
            os.makedirs(dest_folder)
        filename = dest_folder + "/" + name + "_" + key + ".JPG"
        # print(filename)
        cv2.imwrite(filename, augm)


if __name__ == "__main__":
    try:
        assert len(sys.argv) > 1, "missing arguments"

        if sys.argv[1] == "-h":
            print(helper.__doc__)
            exit()

        if len(sys.argv) == 2:
            print(1)
            path = sys.argv[1]
            print(path)
            # Checks if path is a file
            assert os.path.isfile(path), "wrong path"
            print("isFile")
            name = get_img_name(path)
            image_augm(path, name)

#  python Transformation.py -src Apple/apple_healthy/ -dst dst_directory
        elif len(sys.argv) == 5:
            assert sys.argv[1] == "-src" and os.path.isdir(
                sys.argv[2]) and sys.argv[3] == "-dst", "wrong arguments paths"

            path = sys.argv[2]
            print(path)
            name_dir = get_dir_name(path)
            print(name_dir)
            for imgpath in glob.iglob(f'{path}/*'):
                if os.path.isfile(imgpath):
                    name = get_img_name(imgpath)
                    save_image_all_transf(sys.argv[4], imgpath, name)

                else:
                    print("isnotFile:", imgpath)

#  python Transformation.py -src Apple/apple_healthy/ -dst dst_directory -mask
        elif len(sys.argv) == 6:

            assert sys.argv[1] == "-src" and os.path.isdir(
                sys.argv[2]) and sys.argv[3] == "-dst", "wrong arguments paths"
            assert (sys.argv[5] == "-mask" or
                    sys.argv[5] == "-Gblur" or
                    sys.argv[5] == "-histo" or
                    sys.argv[5] == "-landmarks" or
                    sys.argv[5] == "-analyze" or
                    sys.argv[5] == "-keypoints" or
                    sys.argv[5] == "-histo"), helper.__doc__

            path = sys.argv[2]
            print(path)
            name_dir = get_dir_name(path)
            print(name_dir)
            for imgpath in glob.iglob(f'{path}/*'):
                if os.path.isfile(imgpath):
                    name = get_img_name(imgpath)
                    save_image_transf_folder(
                        sys.argv[4], imgpath, name, sys.argv[5])

                else:
                    print("isnotFile:", imgpath)

    except Exception as e:
        print(e)
