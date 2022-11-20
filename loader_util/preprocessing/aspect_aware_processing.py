import numpy as np
import imutils
from cv2 import cv2


class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target dimensions and the interpolation
        # method used during the resizing procedure
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray):
        # grab the dimensions of the image and then initialise
        # the deltas to use when cropping the image
        h, w = image.shape[:2]
        dw = 0
        dh = 0
        # if width is smaller resize along it then update deltas
        # to crop along the other dimension to the desired length
        if w < h:
            image = imutils.resize(image=image, width=self.width)
            dh = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image=image, height=self.height)
            dw = int((image.shape[1] - self.width) / 2.0)

        # now crop using the deltas
        final_image = image[dh:image.shape[0] - dh,
                      dw:image.shape[1] - dw]
        return cv2.resize(final_image,
                          dsize=(self.width, self.height),
                          interpolation=self.inter)
