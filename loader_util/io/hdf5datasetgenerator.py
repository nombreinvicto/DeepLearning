from tensorflow.keras.utils import to_categorical
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None,
                 binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0

        # keep looping infinitle - the model will stop once we have reached
        # desired number of epochs
        while epochs < passes:
            # loop over the dataset
            for i in range(0, self.numImages, self.batchSize):
                # extract images and labels
                images = self.db['images'][i:i + self.batchSize]
                labels = self.db['labels'][i:i + self.batchSize]

                # 1. check to see if the labels shud be binarised
                if self.binarize:
                    labels = to_categorical(labels, self.classes)

                # 2. check to see if there are preprocessors
                if self.preprocessors is not None:
                    # initalise list of processed images
                    procImages = []

                    # loop over the imahes
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        procImages.append(image)
                    images = np.array(procImages)

                # 3. check to see if augmentation provided
                if self.aug is not None:
                    images, labels = next(self.aug.flow(images, labels,
                                                        batch_size=self.batchSize))

                # 4. Finally yield tuple of images and labels
                yield (images, labels)

            # increment epochs
            epochs += 1

    def close(self):
        self.close()
