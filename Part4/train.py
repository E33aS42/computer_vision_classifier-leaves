#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models.vgg import VGG16_Weights
from torchvision import datasets, models, transforms
import time
import os
import splitfolders
import sys
from utils.helperfile import helper_train
from tempfile import TemporaryDirectory


TRAIN = 'train'
VAL = 'val'

# Use NVIDIA Gpu if available
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(dataloaders, model, criterion, optimizer, scheduler,
                num_epochs=5):
    since = time.time()

# 1. Initializing Temporary Storage for Model Checkpoints:
# Creates a temporary directory to store the best model parameters
# during training.
    with TemporaryDirectory() as tmpdir:
        best_model_params_path = os.path.join(tmpdir, 'best_model_params.pt')
# Saves the initial model state (weights, biases) to this path for reference.
        torch.save(model.state_dict(), best_model_params_path)
# Initializes best_acc to track the best validation accuracy achieved so far.
        best_acc = 0.0

# 2. Training Loop for Multiple Epochs:
# Iterates over the specified number of epochs
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

# Iterates through both training and validation phases within each epoch.
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Toggle model to training mode
                else:
                    model.eval()   # Toggle model to evaluate mode

# 3. Within a Phase (Training or Validation):
# Resets running loss and correct predictions for the current phase.
                running_loss = 0.0
                running_corrects = 0

# Iterates over batches of data from the corresponding dataloader.
                for inputs, labels in dataloaders[phase]:
                    # Moves data to the appropriate device (GPU/CPU).
                    inputs = inputs.to(device)
                    labels = labels.to(device)

# Zeroes gradients before each batch update.
                    optimizer.zero_grad()

# Performs a forward pass through the model for prediction
# and loss calculation.
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

# Backpropagates gradients and updates model parameters if in training phase.
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

# Accumulates loss and correct predictions for the phase.
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

# 4. Wrapping Up a Phase:
# Adjusts the learning rate scheduler after the training phase.
# A scheduler is a PyTorch object specifically designed to manage
# the learning rate during the training process of a deep learning model.
                if phase == 'train':
                    scheduler.step()

# Calculates average loss and accuracy for the phase.
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# save the model that performs best on the validation set.
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m \
                {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

# load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


if __name__ == "__main__":
    try:

        assert len(sys.argv) >= 3, helper_train.__doc__

        path = sys.argv[1]
        data_dir = "../train_splitted"
        model_name = sys.argv[2]

# 1. Data Splitting into train-valid-test sets:
        splitfolders.ratio(path, output=data_dir,
                           ratio=(0.8, 0.2, 0.0), seed=42)

# 2. Data Transformations
# VGG-16 Takes 224x224 images as input, so we resize all of them
        data_transforms = {
            TRAIN: transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            VAL: transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        }

# 3. Loading training and validation datasets
        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(data_dir, x),
                transform=data_transforms[x]
            )
            for x in [TRAIN, VAL]
        }

# 4. Creating PyTorch dataloaders for training and validation sets.
# A dataloader efficiently loads batches of data during training.
# We reshuffle the data at every epoch to reduce model overfitting,
# We use up to 8 worker threads for parallel data loading
# to speed up data retrieval
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=8,
                shuffle=True, num_workers=8
            )
            for x in [TRAIN, VAL]
        }

# 5. Calculates and stores the number of images in each phase
        dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL]}

        for x in [TRAIN, VAL]:
            print("Loaded {} images under {}".format(dataset_sizes[x], x))

        print("Classes: ")
        class_names = image_datasets[TRAIN].classes
        print(image_datasets[TRAIN].classes)

# 6. Model creation
# The VGG-16 is able to classify 1000 different labels; we just need 8 instead.
# We are replacing the last fully connected layer
# of the model with a new one with 8 output features instead of 1000.

# 6.1 Loading pretrained VGG16 Model
        # vgg16 = models.vgg16(pretrained=True) # deprecated
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)


# 6.2 Freezing Feature Layers:
# Iteration over the parameters of the features section of the VGG16 model.
# Setting param.require_grad = False freezes the gradients of these parameters.
# Their weights won't be updated during training,
# keeping the pre-trained features intact.
        for param in vgg16.features.parameters():
            param.require_grad = False

# 6.3 Replacing Classifier last layer:
# Gets b of input features from the last layer of the frozen feature extractor.
        num_features = vgg16.classifier[6].in_features
# Remove last layer:
# Creates a list containing all layers from the original classifier
# except the last one.
        features = list(vgg16.classifier.children())[:-1]
# Add a new linear layer with 8 outputs
        features.extend([nn.Linear(num_features, len(class_names))])
# Replace the classifier in the VGG16 model with a new nn.Sequential object
# containing the modified layer
        vgg16.classifier = nn.Sequential(*features)
# summary of the model
        # print(vgg16)

# 7. Moving Model to GPU (if available):
        if use_gpu:
            vgg16.cuda()


# 8. Defining Loss Function and Optimizer:
# Cross-entropy loss measures the performance of
# a classification model whose output is a probability value
# between 0 and 1.
# Cross-entropy loss increases as the predicted probability diverges
# from the actual label.
        criterion = nn.CrossEntropyLoss()

# Creates an optimizer object using stochastic gradient descent with momentum.
        optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

# 9. Learning Rate Scheduler:
# Decays the learning rate of each parameter group by gamma
# every step_size epochs.
# optimizer (Optimizer) – Wrapped optimizer.
# step_size (int) – Period of learning rate decay.
# gamma (float) – Multiplicative factor of learning rate decay.
# Default: 0.1.
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=7, gamma=0.1)

# 10. Model Training

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

        vgg16 = train_model(dataloaders, vgg16, criterion,
                            optimizer_ft, exp_lr_scheduler, num_epochs=2)

        model_scripted = torch.jit.script(vgg16)  # Export to TorchScript
        model_scripted.save(model_name + ".pt")  # Save
    except Exception as e:
        print(e)
