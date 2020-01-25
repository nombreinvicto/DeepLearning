# import the necessary packages
import imutils
import numpy as np
import cv2


def preprocess(image: np.ndarray, width, height):
    # grab the dimensions of the image then init the padding values
    h, w = image.shape[:2]

    # if the width is greater than the height then resize along the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise the height is greater than the weight so resize along height
    else:
        image = imutils.resize(image, height=height)

    # determing padding
    padw = int((width - image.shape[1]) / 2.0)
    padh = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any rounding issue
    image = cv2.copyMakeBorder(image, top=padh, bottom=padh, left=padw,
                               right=padw, borderType=cv2.BORDER_REPLICATE)
    image = cv2.resize(image, dsize=(width, height))

    # return the image
    return image
