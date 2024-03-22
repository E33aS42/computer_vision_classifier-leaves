import numpy as np
import matplotlib.pyplot as plt
import cv2
from plantcv import plantcv as pcv
from utils.mask_ import mask_
# from utils.x_axis_pseudolandmark import x_axis_pseudolandmarks
from utils.y_axis_pseudolandmark import y_axis_pseudolandmarks

"""

Pseudolandmarks on an image don't directly represent real-world features themselves,
but rather act as reference points for computer vision algorithms to analyze and understand the image content.

.What Pseudolandmarks Represent for Algorithms:

- Richer Data Representation: Compared to a few manually marked landmarks, pseudolandmarks provide a denser set of points. This helps the algorithm capture more details about the object or scene in the image.
- Spatial Relationships: The positions of pseudolandmarks relative to each other encode information about the spatial layout of the image content.
- Feature Learning: In tasksmo   like object recognition, the algorithm learns the relationships between these points and specific objects. This allows it to recognize similar objects in new images.

.Applications of Pseudolandmarks:

- Object Recognition: Facial recognition systems might use densely sampled pseudolandmarks across the face to learn the spatial relationships between facial features for identification.
- Object Tracking: In a video sequence, following the changes in positions of pseudolandmarks over time can help track the movement of objects.
- Image Segmentation: Algorithms might use pseudolandmarks to define boundaries between different objects or regions within an image.
- Image Registration: Aligning different images of the same scene can be achieved by establishing correspondences between similar pseudolandmarks across the images.

"""


def Pseudolandmarks(image, ax):
    img = np.copy(image)
    # apply a mask if wanted, otherwise comment this line
    img = mask_(img, "LAB")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # set a sample label name
    pcv.params.sample_label = "plant"

    # Identify a set of land mark points
    # Results in set of point values that may indicate tip points
    left, right, center_h = y_axis_pseudolandmarks(img=img, mask=mask)

    # Access data stored out from y_axis_pseudolandmarks
    left_landmarks = np.asarray(
        pcv.outputs.observations['plant']['left_lmk']['value'])
    right_landmarks = np.asarray(
        pcv.outputs.observations['plant']['right_lmk']['value'])
    center_landmarks = np.asarray(
        pcv.outputs.observations['plant']['center_h_lmk']['value'])

    ax.imshow(image)
    ax.scatter(left_landmarks[:, 0], left_landmarks[:,
               1], marker="o", color="m", s=20)
    ax.scatter(right_landmarks[:, 0],
               right_landmarks[:, 1], marker="o", color="b", s=20)
    ax.scatter(center_landmarks[:, 0], center_landmarks[:,
               1], marker="o", color="#FF6E00", s=20)

    return ax


def Pseudolandmarks_fig(image):
    img = np.copy(image)
    # apply a mask if wanted, otherwise comment this line
    img = mask_(img, "LAB")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # set a sample label name
    pcv.params.sample_label = "plant"
    # Identify a set of land mark points
    # Results in set of point values that may indicate tip points
    left, right, center_h = y_axis_pseudolandmarks(img=img, mask=mask)

    # Access data stored out from y_axis_pseudolandmarks
    left_landmarks = np.asarray(
        pcv.outputs.observations['plant']['left_lmk']['value'])
    right_landmarks = np.asarray(
        pcv.outputs.observations['plant']['right_lmk']['value'])
    center_landmarks = np.asarray(
        pcv.outputs.observations['plant']['center_h_lmk']['value'])

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(left_landmarks[:, 0], left_landmarks[:,
               1], marker="o", color="m", s=20)
    ax.scatter(right_landmarks[:, 0],
               right_landmarks[:, 1], marker="o", color="b", s=20)
    ax.scatter(center_landmarks[:, 0], center_landmarks[:,
               1], marker="o", color="#FF6E00", s=20)

    # Convert the Figure object to a NumPy array so we could later save it into a file
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the image array from RGBA to BGR (OpenCV uses BGR color order)
    image_bgr = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGBA2BGR)
    image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
    return image_bgr
