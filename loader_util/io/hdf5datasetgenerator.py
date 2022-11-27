# import the required packages
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py


# %% ##################################################################
class HDF5DatasetGenerator:
    def __init__(self,
                 db_path,
                 batch_size,
                 preprocessors=None,
                 aug: ImageDataGenerator = None,
                 binarize=True,
                 classes=2):
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.db_path = db_path

        # open the hdf5 db
        self.db = h5py.File(self.db_path, mode="r")
        self.num_images = self.db["labels"].shape[0]

    def generator(self, passes=np.Inf):
        epochs = 0

        while epochs < passes:
            for i in range(0, self.num_images, self.batch_size):
                images = self.db["data"][i: i + self.batch_size]
                labels = self.db["labels"][i: i + self.batch_size]

                # binarize the labels
                if self.binarize:
                    labels = to_categorical(labels,
                                            num_classes=self.classes)
                # preprocess the imahes
                preprocessed_images = []
                if self.preprocessors:
                    for image in images:
                        for preprocessor in self.preprocessors:
                            image = preprocessor.preprocess(image)
                        preprocessed_images.append(image)

                    images = np.array(preprocessed_images)

                # augment the images
                if self.aug:
                    images, labels = next(self.aug.flow(images,
                                                        labels,
                                                        batch_size=self.batch_size))

                # yield the images and labels of current batch
                yield images, labels

            # increment the total number of epochs
            epochs += 1

    def close(self):
        self.db.close()
