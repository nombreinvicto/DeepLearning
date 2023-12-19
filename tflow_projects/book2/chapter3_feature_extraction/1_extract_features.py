import sys
sys.path.append(r"/content/drive/MyDrive")
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from loader_util.io import HDF5DatasetWriter
from tensorflow.keras.models import Model
from imutils import paths
import numpy as np
import logging
import progressbar
import random
import os

# %% ##################################################################
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
# init script constants
data_input_path = r"/content/drive/MyDrive/Colab Notebooks/ImageDataset/book2/animals/images"
outpath = r"/content/drive/MyDrive/Colab Notebooks/ImageDataset/book2/animals/hdf5/features.hdf5"

# batch size is the no of images we pass thru vgg at once
batch_size = 32

# buffer size no of extracted features we store in RAM before flushing
buffer_size = 1000
# %% ##################################################################
logger.info("loading images")
image_paths = list(paths.list_images(data_input_path))
random.shuffle(image_paths)

labels = [pth.split(os.path.sep)[-2] for pth in image_paths]
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
# %% ##################################################################
logger.info("loading VGG network")
model = VGG16(weights="imagenet", include_top=False)  # type: Model
dataset = HDF5DatasetWriter(dims=(len(image_paths), 512 * 7 * 7),
                            outpath=outpath,
                            bufsize=buffer_size)
dataset.store_string_feature_labels(class_labels=le.classes_)
# %% ##################################################################
# initialize the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths),
                               widgets=widgets).start()

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    batch_labels = encoded_labels[i:i + batch_size]
    batch_images = []

    # now iterate over the image paths
    for j, image_path in enumerate(batch_paths):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = imagenet_utils.preprocess_input(img)
        batch_images.append(img)

    # pass processed batches thru CNN
    batch_images = np.vstack(batch_images)
    features = model.predict(batch_images, batch_size=batch_size)
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add the features and labels to hdf5 dataset
    dataset.add(rows=features,
                labels=batch_labels)
    pbar.update(i)

dataset.close()
pbar.finish()
# %% ##################################################################
