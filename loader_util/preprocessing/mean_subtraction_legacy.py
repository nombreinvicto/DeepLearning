import numpy as np
from cv2 import cv2


class MeanSubtractionPreProcessor:
    @staticmethod
    def preprocess(image: np.ndarray):
        B, G, R = cv2.split(image)

        # calculate means
        B_mean = B.mean()
        G_mean = G.mean()
        R_mean = R.mean()

        # calculate std devs
        B_std = B.std()
        G_std = G.std()
        R_std = R.std()

        B = (B - B_mean) / B_std
        G = (G - G_mean) / G_std
        R = (R - R_mean) / R_std

        return cv2.merge((B, G, R))
