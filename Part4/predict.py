#!/usr/bin/env python

import torch
import torch.jit
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import glob
import sys
from utils.helperfile import helper_predict
from PIL import Image
import cv2
from utils.rembg_ import rembg_
from utils.mask_ import mask_


use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dict_prediction = {"good": 0, "bad": 0}


def transform_image(image):
    """
    Transform image and resize if necessary
    """
    image = rembg_(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = mask_(image)
    # image = Pseudolandmarks_fig(image)

    # image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    image = Image.fromarray(image)
    return image


def predict_image(model, image):
    """
    Predict image class
    """
    class_names = [
        "Apple_Black_rot",
        "Apple_healthy",
        "Apple_rust",
        "Apple_scab",
        "Grape_Black_rot",
        "Grape_Esca",
        "Grape_healthy",
        "Grape_spot",
        ]

    # Set the model to evaluation mode
    model.eval()

    # Define image preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Preprocess the image using the defined transformations
    preprocessed_image = transform(image)

    # Add a batch dimension (unsqueeze) as the model expects a batch of images
    preprocessed_image = preprocessed_image.unsqueeze(0)

    # Move the input to the appropriate device (CPU or GPU)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    preprocessed_image = preprocessed_image.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(preprocessed_image)

    # Get the predicted class (assuming the model outputs class probabilities)
    predicted_class = torch.argmax(output, dim=1).item()

    predicted_label = class_names[predicted_class]

    return predicted_label


def is_files_only(directory):
    """
    Checks if a directory contains only files and no subdirectories.

    Args:
        directory: Path to the directory to check.

    Returns:
        True if the directory contains only files, False otherwise.
    """
    try:
        # Get directory contents
        contents = os.listdir(directory)
        # Check if empty (no files or directories)
        if not contents:
            return False

        # Loop through contents and check if all are files
        for item in contents:
            filepath = os.path.join(directory, item)
            if not os.path.isfile(filepath):
                return False
        return True
    except FileNotFoundError:
        # Handle case where directory doesn't exist
        return False


def is_dirs_only(directory):
    """
    Checks if a directory contains only subdirectories and no files.

    Args:
        directory: Path to the directory to check.

    Returns:
        True if the directory contains only subdirectories, False otherwise.
    """
    try:
        # Get directory contents
        contents = os.listdir(directory)
        # Check if empty (no files or directories)
        if not contents:
            return False

        # Loop through contents and check if all are directories
        for item in contents:
            filepath = os.path.join(directory, item)
            if os.path.isfile(filepath):
                return False
        return True
    except FileNotFoundError:
        # Handle case where directory doesn't exist
        return False


def get_label(path):
    '''
    Get image label from path
    '''
    return path.rsplit('/', 1)[0].rsplit('/', 1)[1]


def check_pred(path, prediction):
    '''
    Check if prediction is correct ("good") or not ("bad")
    '''
    label = get_label(path)

    if label == prediction:
        dict_prediction["good"] += 1
    else:
        dict_prediction["bad"] += 1
    # return dict_prediction


def predict_group(model, path):
    """
    get prediction from group of images
    """
    if os.path.isdir(path):
        # check path content
        if is_files_only(path):
            print("files")
    # if content are files
            for imgpath in glob.iglob(f'{path}/*'):
                if os.path.isfile(imgpath):
                    prediction, image, transform_ = get_prediction(
                        imgpath, model)
                    print(imgpath, prediction)
    # if content are directories
        elif is_dirs_only(path):
            print("dirs")
            # print(os.walk(path))
            for root, dirs, files in os.walk(path):
                if root != path:
                    # loop on each directories
                    for imgpath in glob.iglob(f'{root}/*'):
                        #  check if dir content are files
                        if os.path.isfile(imgpath):
                            prediction, image, transform_ = get_prediction(
                                imgpath, model)
                            # print(imgpath, prediction)
                            check_pred(imgpath, prediction)
    #  if yes, make prediction
            print(dict_prediction)
            score = dict_prediction["good"] / \
                (dict_prediction["good"] + dict_prediction["bad"]) * 100
            print(f"Score: {score:.1f}%")

    #  if not return error message and exit()
    else:
        print("no images files in directory")
        exit()


def get_prediction(path, model):
    """
    get prediction from single image
    """
    image = cv2.imread(path)
    transform_ = transform_image(image)
    prediction = predict_image(model, transform_)

    return prediction, image, transform_


def show_prediction(image, transform_, prediction):

    # Create a figure and subplots
    # Adjust figure size as needed
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))

    transform_ = np.array(transform_)

    # Display images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image)
    ax2.imshow(transform_)

    # Turn off axes for cleaner presentation
    ax1.axis('off')
    ax2.axis('off')

    # create formatted subtitle text including prediction
    text = """===             DL classification           ==="""
    fig.text(0.5, 0.2, text, ha='center',
             va='bottom', fontsize=18,
             bbox=dict(facecolor='white', edgecolor='none'))
    fig.text(0.3, 0.1, "Class predicted: ", ha='center',
             va='bottom', fontsize=14, color='black')
    fig.text(0.65, 0.1, prediction, ha='center',
             va='bottom', fontsize=14, color='green')

    # Display the combined plot
    plt.tight_layout()
    plt.show()  # comment this line if running code using Onyxia
    plt.savefig("fig.png")


if __name__ == "__main__":

    try:
        assert len(sys.argv) >= 3, helper_predict.__doc__

        # Load the model from the .pt file using torch.load
        model = torch.jit.load(sys.argv[2])

        # path the image(s) to predict
        path = sys.argv[1]

        if os.path.isfile(path):
            prediction, image, transform_ = get_prediction(path, model)
            show_prediction(image, transform_, prediction)
        else:
            predict_group(model, path)

    except Exception as e:
        print(e)
