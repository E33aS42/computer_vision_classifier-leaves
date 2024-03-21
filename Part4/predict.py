import torch
import torch.jit
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import glob
import copy
import sys
from PIL import Image
import cv2
from utils.rembg_ import rembg_
from utils.mask_ import mask_
from utils.pseudolandmarks import Pseudolandmarks_fig


#  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
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
    class_names = ["Apple_Black_rot", "Apple_healthy", "Apple_rust", "Apple_scab", \
                   "Grape_Black_rot", "Grape_Esca", "Grape_healthy", "Grape_spot"]
    
    # Set the model to evaluation mode
    model.eval()

    # Define image preprocessing steps
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # transform = transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #         ])

    # Preprocess the image using the defined transformations
    # image = Image.fromarray(image)
    preprocessed_image = transform(image)

    # Add a batch dimension (unsqueeze) as the model expects a batch of images
    preprocessed_image = preprocessed_image.unsqueeze(0)

    # # Move the input to the appropriate device (CPU or GPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    preprocessed_image = preprocessed_image.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(preprocessed_image)

    # Get the predicted class (assuming the model outputs class probabilities)
    predicted_class = torch.argmax(output, dim=1).item()

    predicted_label = class_names[predicted_class]

    # print(f"Predicted class: {predicted_label}")

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
    if os.path.isdir(path):
    # check path content
        if is_files_only(path):
            print("files")
    # if content are files
            for imgpath in glob.iglob(f'{path}/*'):
                if os.path.isfile(imgpath):
                    prediction, image, transform_ = get_prediction(imgpath, model)
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
                            prediction, image, transform_ = get_prediction(imgpath, model)
                            # print(imgpath, prediction)
                            check_pred(imgpath, prediction)

            print(dict_prediction)
            score = dict_prediction["good"] / (dict_prediction["good"] + dict_prediction["bad"]) * 100
            print(f"Score: {score:.1f}%")

    #  if not return error message and exit()
    #  if yes, make prediction and store it into text file with file path

    else:
        exit()

def predict_group_cuda(dataloaders, model, criterion, optimizer, scheduler,
                 num_epochs=5):
    since = time.time()

    model.eval()
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

def get_prediction(path, model):
    
    image = cv2.imread(path)

    transform_ = transform_image(image)
    # plt.imshow(transform, cmap='gray')
    # plt.show()

    prediction = predict_image(model, transform_)

    return prediction, image, transform_ 

def show_prediction(image, transform_, prediction):

    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))  # Adjust figure size as needed

    transform_ = np.array(transform_)
    # transform_ = cv2.cvtColor(transform_, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image)
    # Display images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image)
    ax2.imshow(transform_)

    # Turn off axes for cleaner presentation
    ax1.axis('off')
    ax2.axis('off')

    # create formatted subtitle text including prediction
    fig.text(0.5, 0.2, "===             DL classification           ===", ha='center', va='bottom', fontsize=18, bbox=dict(facecolor='white', edgecolor='none'))
    fig.text(0.3, 0.1, "Class predicted: ", ha='center', va='bottom', fontsize=14, color='black')
    fig.text(0.65, 0.1, prediction, ha='center', va='bottom', fontsize=14, color='green')

    # Display the combined plot
    plt.tight_layout()
    plt.show() # comment this line if running code using Onyxia
    plt.savefig("fig.png")


if __name__ == "__main__":

    # if len(sys.argv) >= 2:
    
    # Load the model from the .pt file using torch.load
    # model = torch.load(sys.argv[2])
    # model = torch.jit.load("model_scripted_8_mask_60.pt")
    # model = torch.jit.load("model_scripted_8_origin.pt")
    # model = torch.jit.load("model_scripted_8_landmarks.pt")
    model = torch.jit.load("model_scripted1.pt")

    path = sys.argv[1]

    if os.path.isfile(path):
        prediction, image, transform_ = get_prediction(path, model)
        show_prediction(image, transform_, prediction)
    else:
        predict_group(model, path)
    


