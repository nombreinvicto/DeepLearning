# import the required packages
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Model
import numpy as np
import imutils


def sliding_window(image: np.ndarray, step: int, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield x, y, image[y:y + ws[1], x:x + ws[0]]


def image_pyramid(image: np.ndarray, scale=1e-5, min_size=(224, 224)):
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image doesnt meet the supplied min size, then stop
        # constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield image


