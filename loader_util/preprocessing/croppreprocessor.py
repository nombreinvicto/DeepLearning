from collections import namedtuple
import numpy as np
import cv2

Coords = namedtuple("Coords", "startx starty endx endy")


class CropPreprocessor:
    def __init__(self,
                 width,
                 height,
                 horiz=True,
                 inter=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image: np.ndarray):
        crops = []

        # grab the dims of the bigger image
        h, w = image.shape[:2]
        coords_set = [
            # this is for the upper left crop
            Coords(startx=0,
                   starty=0,
                   endx=self.width,
                   endy=self.height),
            # this is for the upper right crops
            Coords(startx=w - self.width,
                   starty=0,
                   endx=w,
                   endy=self.height),
            # this is for the bottom left crop
            Coords(startx=0,
                   starty=h - self.height,
                   endx=self.width,
                   endy=h),
            # this is for the bottom right crop
            Coords(startx=w - self.width,
                   starty=h - self.height,
                   endx=w,
                   endy=h)
        ]

        # compute the center crop
        dw = int(0.5 * (w - self.width))
        dh = int(0.5 * (w - self.height))

        coords_set.append(
            Coords(startx=dw, starty=dh, endx=w - dw, endy=h - dh)
        )

        # now loop over the coords and extract crops
        for coord in coords_set:
            crop = image[coord.starty:coord.endy, coord.startx:coord.endx]
            crop = cv2.resize(crop,
                              dsize=(self.width, self.height),
                              interpolation=self.inter)
            crops.append(crop)

        # finally take flips
        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        # finally return array of crops
        return np.array(crops)
