# computer_vision_classifier-leaves
Computer Vision classifier using a VGG16 pretrained model on an augmented and transformed dataset of diseased leaves images.

This is a school project that was divided into 4 parts

# Part 1 : Data analysis

The dataset consists of 8 classes of apple and grape tree leaves, 1 healthy class and 3 diseases classes for each fruit tree.

Looking at the data shows that it is clearly unbalanced.

![pie](https://github.com/E33aS42/computer_vision_classifier-leaves/assets/66993020/b3bdfbaa-d3bf-4e56-8c79-0b1eeeafebad)

# Part 2 : Dataset augmentation and balancing

In order to balance the dataset, new images were created from the existing images using different types of images augmentations.

![augm](https://github.com/E33aS42/computer_vision_classifier-leaves/assets/66993020/ca6582ec-8d0b-4e6f-97cd-a52b2a1436e3)


![pie_balanced](https://github.com/E33aS42/computer_vision_classifier-leaves/assets/66993020/cf912c2d-c636-472f-be4e-a7c945dc84e9)

Note: All image backgrounds have been replaced with a black background.

# Part 3 : Images characteristics extraction

Different methods of direct extraction of characteristics from an image of a leaf were implemented.

- Applying a Gaussian blur filter to reduce noise or softening harsh edges.

- Applying a mask to isolate or remove objects in an image.

- Analyze the size and shape characteristics of the leaf.

- Drawing pseudo-landmarks on an image to study the leaf morphology and shape and to learn the spatial relationships between features. Those landmarks can work as reference points for computer vision algorithms to analyze and understand the image content.

- Detecting keypoints or points of interests within an image to represent locations such as corners, edges, or blobs. Potential keypoints are identified based on intensity differences between a central pixel and its surrounding pixels. An orientation is assigned based on the intensity distribution around the keypoint making them rotation invariant. This is useful for image matching and object recognition.

- Drawing the color histogram of an image to show the distribution of colors within that image. It tells us how many pixels fall into each color range and it is a valuable tool for understanding the color characteristics of an image.

![transf4](https://github.com/E33aS42/computer_vision_classifier-leaves/assets/66993020/a9e9a97c-2525-49db-8bdf-24d189779ad9)

![color_hist5](https://github.com/E33aS42/computer_vision_classifier-leaves/assets/66993020/577d9e14-4abb-41ca-aa94-de2390c46fd0)


# Part 4 : Model training and prediction

## Model setup

The pre-trained VGG16 model was used to build the leaf disease prediction model.

- The data is split into three datasets: training, validation, and test, for cross-validation.
- Images must be resized since VGG16 takes 224×224 images as input.
- Creation of PyTorch dataloaders to allow for automatic batch loading of data from the training and validation datasets during training.
- Data is shuffled at each epoch to reduce overfitting.
- Eight worker threads are used to enable parallel data loading and speed up the process.
- VGG-16 is capable of classifying 1,000 different classes. Since only 8 classes need to be classified, the gradients of the different layers of the model are frozen so that the feature weights are not updated during training, keeping the pre-trained features intact. Then the last layer (the fully connected layer) of the model is replaced with a new layer containing only 8 features.

## Model training

For every epoch we iterate over all the training batches, compute the loss, and adjust the network weight. Then we evaluate the performance over the validaton set. At the end of every epoch we print the network progress (loss and accuracy). The accuracy tells us how many predictions were correct.

Training the model during 5 epochs has been found to be enough to get high accuracy without showing overfitting.

## Model prediction

The trained model has been tested on the test dataset and presented an accuracy rate superior to 98%.

Example of a Test sample predition:

![fig](https://github.com/E33aS42/computer_vision_classifier-leaves/assets/66993020/a50e4fe7-5607-4b66-b758-b3770caac701)
