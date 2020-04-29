import numpy as np
from cv2 import cv2


class MeanSubtractionPreProcessor:
    @staticmethod
    def preprocess(image: np.ndarray):
        B, G, R = cv2.split(image)
        epsillon = 1e-4

        # calculate means
        B_mean = B.mean()
        G_mean = G.mean()
        R_mean = R.mean()

        # calculate std devs
        B_std = B.std()
        G_std = G.std()
        R_std = R.std()

        B = (B - B_mean) / (B_std + epsillon)
        G = (G - G_mean) / (G_std + epsillon)
        R = (R - R_mean) / (R_std + epsillon)

        return cv2.merge((B, G, R))
