# import the necessary packages
import imutils
import numpy as np
from cv2 import cv2


class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        dw = 0
        dh = 0

        # if width is smaller than height, then resize along the width
        if w < h:
            # imutils resize does an aspect aware resize
            image = imutils.resize(image, width=self.width)
            dh = int((image.shape[0] - self.height) / 2.0)

        else:
            image = imutils.resize(image, height=self.height)
            dw = int((image.shape[1] - self.width) / 2.0)

        # now that images have been resized, re-grab width height then crop
        h, w = image.shape[:2]
        image = image[dh:h - dh, dw:w - dw]

        # finally resize the image to provide spatial dimensions to ensure
        # output is always a fixed size
        return cv2.resize(image, dsize=(self.width, self.height),
                          interpolation=self.inter)
