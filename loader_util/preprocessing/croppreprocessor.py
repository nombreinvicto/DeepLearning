import cv2
import numpy as np


class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def proprocess(self, image: np.ndarray):
        h, w = image.shape[:2]

        coords = [
            # starty, endy, startx, endx
            (0, self.height, 0, self.width),
            (0, self.height, w - self.width, w),
            (h - self.height, h, 0, self.width),
            (h - self.height, h, w - self.width, w),
        ]

        # coords for center crop
        dw = int(0.5 * (w - self.width))
        dh = int(0.5 * (h - self.height))
        coords.append((dh, h - dh, dw, w - dw))

        crops = []
        for (starty, endy, startx, endx) in coords:
            crop = image[starty:endy, startx:endx]
            crop = cv2.resize(crop, (self.width, self.height),
                              interpolation=self.inter)
            crops.append(crop)

        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)
