# import the necessary packages
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image: np.ndarray):
        return extract_patches_2d(image=image,
                                  patch_size=(self.height, self.width),
                                  max_patches=1)[0]
