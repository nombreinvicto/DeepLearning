# import the necessary packages
import cv2
import numpy as np


class MeanPreprocessor:
    def __init__(self, rmean, gmean, bmean):
        self.rmean = rmean
        self.gmean = gmean
        self.bmean = bmean

    def preprocess(self, image: np.ndarray):
        B, G, R = cv2.split(image.astype("float32"))

        # subtract the means
        B -= self.bmean
        G -= self.gmean
        R -= self.rmean

        # merge the channels back together
        return cv2.merge([B, G, R])
