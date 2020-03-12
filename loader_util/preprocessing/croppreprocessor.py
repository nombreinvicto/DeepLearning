import numpy as np
from cv2 import cv2


class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):
        # initialise the list of crops
        crops = []

        height, width = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [width - self.width, 0, width, self.height],
            [width - self.width, height - self.height, width, height],
            [0, height - self.height, self.width, height]
        ]

        # also compute the center crop
        dw = int(0.5 * (width - self.width))
        dh = int(0.5 * (height - self.height))
        coords.append(([dw, dh, width - dw, height - dh]))

        # loop over the coords and extract crops
        for startx, starty, endx, endy in coords:
            crop = image[starty:endy, startx:endx]
            crop = cv2.resize(crop,
                              dsize=(self.width, self.height),
                              interpolation=self.inter)
            crops.append(crop)

        # check to see if horiz is True
        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)
