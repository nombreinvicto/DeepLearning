# import necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from loader_util.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import random
import os

# %%

dataset = r"C:\Users\mhasa\Google Drive\Tutorial " \
          r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\all_cats_dogs\train"
output = r"C:\Users\mhasa\Google Drive\Tutorial " \
         r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets" \
         r"\all_cats_dogs\hdf5\extracted_features.hdf5"
batch_size = 16
buffer_size = 1000
# %%

print(f"[INFO] loading images......")
image_paths = list(paths.list_images(dataset))
random.shuffle(image_paths)
# %%

# extract the class labels
labels = [pt.split(os.path.sep)[-1].split(".")[0] for pt in image_paths]
unique_labels = np.unique(labels)

# encode the labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
# %%

print(f"[INFO] loading network......")
model = ResNet50(weights="imagenet", include_top=False)  # type: Model

# init the dataset
dataset_writer = HDF5DatasetWriter(dims=(len(image_paths), 7 * 7 * 2048),
                                   outputPath=output,
                                   bufSize=buffer_size)
# %%

# init the progressbar
# initialize the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths),
                               widgets=widgets).start()

# loop over images in batches
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    batch_labels = encoded_labels[i:i + batch_size]
    batch_images = []

    # loop over images in a single batch
    for j, image_path in enumerate(batch_paths):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        batch_images.append(image)

    # pass batches thru the model
    batch_images = np.vstack(batch_images)
    features = model.predict(batch_images, batch_size=batch_size)
    features = features.reshape(features.shape[0], -1)

    # add the batch of features to the dataset
    dataset_writer.add(features, batch_labels)
    pbar.update(i)

# close the dataset
pbar.finish()
#%%