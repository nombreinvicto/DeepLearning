# import the needed packages
import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor, init to [] if None
        self.preprocessors = preprocessors or []

    def load(self, imagePaths, verbose=1):
        # init the list of feature and labels
        data = []
        labels = []

        # loop over the input images
        for i, imagePath in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # path has /path/to/dataset/{class}/{image}.jpg format
            image = cv2.imread(imagePath)
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
                print(f"[INFO] processed {i + 1}/{len(imagePaths)}")

        return np.array(data), np.array(labels)
