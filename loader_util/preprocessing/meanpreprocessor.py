from cv2 import cv2


class MeanPreprocessor:
    def __init__(self, rmean, gmean, bmean):
        # store the Red, Green and Blue channel average aceoss training set
        self.rmean = rmean
        self.gmean = gmean
        self.bmean = bmean

    def preprocess(self, image):
        # split the image into channels
        B, G, R = cv2.split(image.astype('float32'))

        # subtract the means for each channel
        R -= self.rmean
        G -= self.gmean
        B -= self.bmean

        # merge the channels back together and return the image
        return cv2.merge([B, G, R])
