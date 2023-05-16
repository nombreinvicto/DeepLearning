# converts PIL RGB to numpy array with correct dataformat or channel ordering
from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply keras utility that correctly rearranged dimensions of image
        return img_to_array(image, data_format=self.dataFormat)
