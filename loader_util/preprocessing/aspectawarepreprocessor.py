import cv2
import imutils
import numpy as np


class AspectAwarePreprocessor:
    def __init__(self,
                 width,
                 height,
                 inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        dw, dh = 0, 0

        # determine shortest side and resize along that side
        if w < h:
            # resize along width
            image = imutils.resize(image=image,
                                   width=self.width,
                                   inter=self.inter)

            # returned image is a resized one via aspect-aware
            # only thing left is cropping along the longer axis
            # to match the desired dimension
            dh = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image=image,
                                   height=self.height,
                                   inter=self.inter)
            dw = int((image.shape[1] - self.width) / 2.0)

        # grab the aspect aware resized (but not cropped) dims
        ha, wa = image.shape[:2]

        # finally crop along the longer axis
        image = image[dh:h - dh, dw:w - dw]

        return cv2.resize(image,
                          (self.width, self.height),
                          interpolation=self.inter)
