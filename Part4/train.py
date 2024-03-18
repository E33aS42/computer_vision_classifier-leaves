"""

install split-folders into environment:
pip install split-folders

"""


import torch
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
import splitfolders
from tempfile import TemporaryDirectory


path = "../train_directory"
data_dir = "../train_splitted"
TRAIN = 'train'
VAL = 'val'

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/
# https://fmorenovr.medium.com/set-up-conda-environment-pytorch-1-7-cuda-11-1-96a8e93014cc


def train_model1(dataloaders, model, criterion, optimizer, scheduler,
                 num_epochs=5):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m \
                {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


if __name__ == "__main__":
    try:

        # 1. split data into train-valid-test sets:
        splitfolders.ratio(path, output=data_dir,
                           ratio=(0.8, 0.2, 0.0), seed=42)

        # 2. resize images
        # VGG-16 Takes 224x224 images as input, so we resize all of them
        data_transforms = {
            TRAIN: transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            VAL: transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        }

# ImageFolder helps us to easily create PyTorch training
# and validation datasets without writing custom classes.
# Then we can use these datasets to create our iterable data loaders.
        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(data_dir, x),
                transform=data_transforms[x]
            )
            for x in [TRAIN, VAL]
        }
# The Dataset retrieves our dataset's features and labels one sample at a time.
# While training a model, we typically want to pass samples in “minibatches”,
# reshuffle the data at every epoch to reduce model overfitting,
# and use Python’s multiprocessing to speed up data retrieval.
# Setting the argument num_workers as a positive integer will turn on
# multi-process data loading with the specified number
# of loader worker processes.
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=8,
                shuffle=True, num_workers=8
            )
            for x in [TRAIN, VAL]
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL]}

        for x in [TRAIN, VAL]:
            print("Loaded {} images under {}".format(dataset_sizes[x], x))

        print("Classes: ")
        class_names = image_datasets[TRAIN].classes
        print(image_datasets[TRAIN].classes)

        # 3. Model creation
# The VGG-16 is able to classify 1000 different labels; we just need 8 instead.
# In order to do that we are going replace the last fully connected layer
# of the model with a new one with 8 output features instead of 1000.

# In PyTorch, we can access the VGG-16 classifier with model.classifier,
# which is an 6-layer array. We will replace the last entry.

# We can also disable training for the convolutional layers setting
# require_grad = False, as we will only train the fully connected classifier.

        # Load the pretrained model from pytorch
        vgg16 = models.vgg16(pretrained=True)
        # vgg16.load_state_dict(torch.load("vgg16-397923af.pth"))
        # Loads the schedulers state.
        print(vgg16.classifier[6].out_features)

# Freeze training for all layers:
# every parameter has an attribute called requires_grad
# which is by default True.
# True means it will be backpropagrated and hence to freeze a layer
# you need to set requires_grad to False for all parameters of a layer.
        for param in vgg16.features.parameters():
            param.require_grad = False

        # Newly created modules have require_grad=True by default
        num_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1]  # Remove last layer
        # Add our layer with 8 outputs
        features.extend([nn.Linear(num_features, len(class_names))])
        # Replace the model classifier
        vgg16.classifier = nn.Sequential(*features)
        # print(vgg16) # summary of the model

        if use_gpu:
            vgg16.cuda()  # .cuda() will move everything to the GPU side

        resume_training = False

        if resume_training:
            print("Loading pretrained model..")
            vgg16.load_state_dict(torch.load('VGG16-leafaffliction.pt'))
            print("Loaded!")

# define models parameters: loss function (cross entropy) and optimizer.
# Cross-entropy loss, or log loss, measures the performance of
# a classification model whose output is a probability value
# between 0 and 1.
# Cross-entropy loss increases as the predicted probability diverges
# from the actual label.
        criterion = nn.CrossEntropyLoss()

    # Implements stochastic gradient descent (optionally with momentum).
        optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

#  Decays the learning rate of each parameter group by gamma
# every step_size epochs.
# optimizer (Optimizer) – Wrapped optimizer.
# step_size (int) – Period of learning rate decay.
# gamma (float) – Multiplicative factor of learning rate decay.
# Default: 0.1.
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=7, gamma=0.1)

        # 4. model Training

# For every epoch we iterate over all the training batches, compute the loss ,
# and adjust the network weights with loss.backward() and optimizer.step().
# Then we evaluate the performance over the validaton set.
# At the end of every epoch we print the network progress (loss and accuracy).
# The accuracy will tell us how many predictions were correct.

# As we said before, transfer learning can work on smaller dataset too,
# so for every epoch we only iterate over half the trainig dataset
# (worth noting that it won't exactly be half of it over the entire training,
# as the data is shuffled, but it will almost certainly be a subset)

        print(torch.cuda.is_available())
        print("Pytorch CUDA Version is ", torch.version.cuda)

        vgg16 = train_model1(dataloaders, vgg16, criterion,
                             optimizer_ft, exp_lr_scheduler, num_epochs=2)

        model_scripted = torch.jit.script(vgg16)  # Export to TorchScript
        model_scripted.save('model_scripted_8.pt')  # Save
    except Exception as e:
        print(e)
