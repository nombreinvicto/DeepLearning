# import the necessary packages
import random
import sys, os
import logging
import numpy as np
from imutils import paths
import argparse, progressbar
from loader_util.io import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# %% ##################################################################
# script constants
dataset_path = r"C:\Users\mhasa\google_drive\PYTH\DeepLearning\DeepLearning-DL4CV\Edition3\datasets\animals\images"
output = r"C:\Users\mhasa\google_drive\PYTH\DeepLearning\DeepLearning-DL4CV\Edition3\datasets\animals\hdf5\features.hdf5"
batch_size = 32
# %% ##################################################################
# init logging properties
logging_name = "extract_features"
logger = logging.getLogger(logging_name)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s "
                              "%(levelname)s: "
                              "%(name)s: -> %(message)s",
                              "%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# %% ##################################################################
logger.info(f"loading images")
image_paths = list(paths.list_images(dataset_path))
random.shuffle(image_paths)
labels = [path.split(os.path.sep)[-2] for path in image_paths]

# init the label encoder
le = LabelEncoder()
labels = le.fit_transform(labels)
# %% ##################################################################
logger.info("loading network")
model = VGG16(weights="imagenet", include_top=False)

# init the HDF5 dataset writer and store class label names
dataset = HDF5DatasetWriter(dims=(len(image_paths), 512 * 7 * 7),
                            output_path=output,
                            data_key="features")
dataset.store_class_labels(le.classes_)
# %% ##################################################################
# initialize the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths),
                               widgets=widgets).start()

# loop over images in batches
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    batch_labels = labels[i:i + batch_size]
    batch_images = []

    # loop over images and labels of current batch
    for j, image_path in enumerate(batch_paths):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        batch_images.append(image)

    # pass the batch if images thru the network
    batch_images = np.vstack(batch_images)
    features = model.predict(batch_images, batch_size=batch_size)
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add the features amtrix into the hdf5 dataset
    dataset.add(features, batch_labels)
    pbar.update(i)

# finish everything
dataset.close()
pbar.finish()
# %% ##################################################################
