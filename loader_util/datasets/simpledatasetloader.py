# import the needed packages
import numpy as np
from cv2 import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor, init to [] if None
        self.preprocessors = preprocessors or []

    def load(self, image_paths, verbose=1):
        # init the list of feature and labels
        data = []
        labels = []

        # loop over the input images
        for i, imagePath in enumerate(image_paths):
            # load the image and extract the class label assuming
            # path has /path/to/dataset/{class}/{image}.jpg format
            # print(imagePath)
            image = cv2.imread(imagePath)
            image = image.astype('float')
            label = imagePath.split(os.path.sep)[-2]

            # check to see if preprocessors are not None. Then loop
            # over the preprocessors and apply each to image
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    # in chapter7 image is now a 32x32x3 numpy matrix

            data.append(image)
            labels.append(label)

            # show an update every 'verbose' image
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processed {i + 1}/{len(image_paths)}")

        return np.array(data), np.array(labels)
