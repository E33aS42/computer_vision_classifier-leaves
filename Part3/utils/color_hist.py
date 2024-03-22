
"""
An image histogram is a graphical representation of the number of pixels
in an image as a function of their intensity.

An histogram is a graph showing the number of pixels in an image
at each different intensity value found in that image.
Histograms are made up of bins, each bin representing a certain intensity value range.
The histogram is computed by examining all pixels in the image and assigning each
to a bin depending on the pixel intensity. The final value of a bin is the number
of pixels assigned to it. The number of bins in which the whole intensity range
is divided is usually in the order of the square root of the number of pixels.

Image histograms are an important tool for inspecting images.
They allow you to spot BackGround (Background is any additive and approximately
constant signal in your image that is not coming from the objects you are interested in.
Background might come from for example an electronic offset in your detector
or from indirect light) and grey value range at a glance.


"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# RGB Color Space:
# Red, blue and green are the primary colors, each of which is visible to the human eye.
# Visible colors are considered to be combinations of these three.

def color_hist(image):
    # split the image into its respective channels, then initialize the
    # tuple of channel names along with our figure for plotting
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chans = cv2.split(image)
    colors = ("b", "g", "r")  # cv2 uses BGR
    labels = ("blue", "green", "red", "hue", "saturation",
              "value/brightness", "lightness", "green-magenta", "blue-yellow")
    fig = plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Pixels intensity")
    plt.ylabel("Proportion of Pixels (%)")

    totalpixels = 3 * 256 * 256

    # loop over the image channels
    i = 0
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist / totalpixels * 100
        plt.plot(hist, color=color, label=labels[i])
        plt.xlim([0, 256])
        i += 1

    # The HSV (Hue, Saturation, Value) model defines a color in terms of
    # three constituent components:
    # Hue, the color type (such as red, blue, or yellow),
    # Saturation, the "vibrancy" or "purity" of the color,
    # Value, the brightness of the color.

    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    chans_hsv = cv2.split(img_HSV)
    colors_hsv = ("m", "c", "k")

    for (chan, color) in zip(chans_hsv, colors_hsv):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist / totalpixels * 100
        plt.plot(hist, color=color, label=labels[i])
        plt.xlim([0, 256])
        i += 1

    # L is for lightness. It goes from 0 to 100.
    # Lightness shows the contrast between black and grays in a picture.
    # a is red to green. The negative axis is green and the positive is red.
    # b goes from yellow to blue. Blue lies on the negative side
    # and yellow on the positive one.

    img_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    chans_lab = cv2.split(img_LAB)
    colors_lab = ("grey", "pink", "y")

    for (chan, color) in zip(chans_lab, colors_lab):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist / totalpixels * 100
        plt.plot(hist, color=color, label=labels[i])
        plt.xlim([0, 256])
        i += 1

    # Transform your image from RGB to HSV color space using cvtColor()
    # with CV_BGR2HSV or CV_RGB2HSV option.
    # H, S and V stands for Hue, Saturation and Intensity respectively.
    plt.legend(title='Color Channels')
    plt.grid()
    plt.show()

    # Convert the Figure object to a NumPy array so we could later
    # save it into a file
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the image array from RGBA to BGR (OpenCV uses BGR color order)
    image_bgr = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGBA2BGR)

    return image_bgr


def color_hist_save(image):
    # split the image into its respective channels, then initialize the
    # tuple of channel names along with our figure for plotting
    chans = cv2.split(image)
    colors = ("b", "g", "r")  # cv2 uses BGR
    labels = ("blue", "green", "red", "hue", "saturation",
              "value/brightness", "lightness", "green-magenta", "blue-yellow")
    fig = plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Pixels intensity")
    plt.ylabel("Proportion of Pixels (%)")

    totalpixels = 3 * 256 * 256

    # loop over the image channels
    i = 0
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist / totalpixels * 100
        plt.plot(hist, color=color, label=labels[i])
        plt.xlim([0, 256])
        i += 1

    # The HSV (Hue, Saturation, Value) model defines a color in terms
    # of three constituent components:
    # Hue, the color type (such as red, blue, or yellow),
    # Saturation, the "vibrancy" or "purity" of the color,
    # Value, the brightness of the color.

    # HSV is a color space designed to mimic how humans perceive color.
    # It separates color information into three intuitive components:

    # Hue (H): Represents the actual color itself, like red, green, or blue.
    # It's typically visualized as a circular spectrum with values ranging
    # from 0 to 360 degrees (0° for red, 120° for green, 240° for blue, and so on).

    # Saturation (S): Represents the intensity or purity of the color.
    # A saturation of 0% results in a gray tone, while 100% is a fully saturated color.

    # Value (V): Represents the overall brightness of the color,
    # ranging from 0% (black) to 100% (white).

    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    chans_hsv = cv2.split(img_HSV)
    colors_hsv = ("m", "c", "k")

    for (chan, color) in zip(chans_hsv, colors_hsv):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist / totalpixels * 100
        plt.plot(hist, color=color, label=labels[i])
        plt.xlim([0, 256])
        i += 1

    # L is for lightness. It goes from 0 to 100. Lightness shows
    # the contrast between black and grays in a picture.
    # a is red to green. The negative axis is green and the positive is red.
    # b goes from yellow to blue. Blue lies on the negative side
    # and yellow on the positive one.

    img_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    chans_lab = cv2.split(img_LAB)
    colors_lab = ("grey", "pink", "y")

    for (chan, color) in zip(chans_lab, colors_lab):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist / totalpixels * 100
        plt.plot(hist, color=color, label=labels[i])
        plt.xlim([0, 256])
        i += 1

    # Transform your image from RGB to HSV color space using cvtColor()
    # with CV_BGR2HSV or CV_RGB2HSV option.
    # H, S and V stands for Hue, Saturation and Intensity respectively.
    plt.legend(title='Color Channels')
    plt.grid()

    # Convert the Figure object to a NumPy array so we could later save it into a file
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Convert the image array from RGBA to BGR (OpenCV uses BGR color order)
    image_bgr = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGBA2BGR)

    return image_bgr
