# from google.colab import drive
#
# drive.mount('/content/drive')
# %%
import random
import sys, os
import numpy as np
import argparse, progressbar
from cv2 import cv2
from tensorflow.keras.applications import imagenet_utils

# %%
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from loader_util.io import HDF5DatasetWriter
##
from tensorflow.keras.preprocessing.image import ImageDataGenerator, \
    img_to_array, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from imutils import paths

# %%

args = {
    "dataset": r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\animals\images",
    "output": r"C:\Users\mhasa\Google Drive\Tutorial "
              r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets"
              r"\animals\hdf5\extracted_features.hdf5",
    "batch_size": 32,
    "buffer_size": 1000
}
# %%
print(f"[INFO] loading images......")
batch_size = args["batch_size"]
image_paths = list(paths.list_images(args["dataset"]))
random.shuffle(image_paths)

# encode the labels
le = LabelEncoder()
class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
encoded_labels = le.fit_transform(class_names)
# %%

print(f"[INFO] loading pretrained network......")
model = VGG16(weights="imagenet", include_top=False)  # type: Model

dataset = HDF5DatasetWriter(dims=(len(image_paths), 512 * 7 * 7),
                            outputPath=args["output"],
                            bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)
# %%

# init the progressbar
widgets = [f"Extracting Features: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths),
                               widgets=widgets).start()

# loop over images in batches
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    batch_labels = encoded_labels[i:i + batch_size]
    batch_images = []

    # loop over images and labels in the current batch
    for j, image_path in enumerate(batch_paths):
        #image = load_img(image_path, target_size=(224, 224))
        image = cv2.imread(image_path) # type: np.ndarray
        image = cv2.resize(image, dsize=(224, 224))
        image = img_to_array(image)

        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        batch_images.append(image)

    # pass batch of images thru net
    batch_images = np.vstack(batch_images)
    extracted_features = model.predict(batch_images)
    extracted_features = \
        extracted_features.reshape((extracted_features.shape[0], -1))

    # add to hdf5 dataset
    dataset.add(extracted_features, batch_labels)
    pbar.update(i)

pbar.finish()
dataset.close()
# %%
