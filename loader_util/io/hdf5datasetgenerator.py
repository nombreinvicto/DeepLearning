from tensorflow.keras.preprocessing.image import ImageDataGenerator
from loader_util.preprocessing import SimplePreProcessor
from tensorflow.keras.utils import to_categorical
from typing import Dict, Sequence
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self,
                 dbpath,
                 batch_size,  # size of batch to yield
                 preprocessors: Sequence[SimplePreProcessor] = None,
                 aug: ImageDataGenerator = None,
                 binarize=True,  # whether to OHE the ordinal labels
                 features_name="data",
                 labels_name="labels"):
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.features_name = features_name
        self.labels_name = labels_name

        # open the HDF5 db
        self.db = h5py.File(dbpath, mode="r")
        self.num_images = self.db[self.features_name].shape[0]
        self.classes = len(np.unique(self.db[self.labels_name]))
        print(f"[INFO] initialised HDF5 generator with {self.num_images} "
              f"data across {self.classes} classes")

    def generator(self, passes=np.inf):
        epochs = 0

        # keep looping infinitely, model will stop
        # once desired epoch numbers is reached
        while epochs < passes:
            for i in range(0, self.num_images, self.batch_size):
                # handle a batch of data at a time
                images = self.db[self.features_name][i:i + self.batch_size]
                labels = self.db[self.labels_name][i:i + self.batch_size]

                # OHE the labels
                if self.binarize:
                    labels = to_categorical(labels,
                                            num_classes=self.classes)

                # use preprocessors if needed
                if self.preprocessors:
                    proc_images = []
                    for image in images:
                        for preprocessor in self.preprocessors:
                            image = preprocessor.preprocess(image)
                        proc_images.append(image)
                    images = np.array(proc_images)

                # use augmentors if present
                # flow() Returns
                # An `Iterator` yielding tuples of `(x, y)`
                # where `x` is a NumPy array of image data
                # and `y` is a NumPy array of corresponding labels.
                if self.aug:
                    images, labels = next(self.aug.flow(images, labels,
                                                        batch_size=self.batch_size))

                # finally yield tuples of images and labels
                yield (images, labels)

            # increment the epoch after all batches processed
            epochs += 1

    def close(self):
        self.db.close()
