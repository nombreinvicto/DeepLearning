from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from loader_util.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import random
import os

# %% ##################################################################
# script constants
dataset_path = r"C:\Users\mhasa\Downloads\delete\train\train"
extracted_features_output = r"C:\Users\mhasa\PycharmProjects\deep_learning\loader_util\datasets\kaggle_dogs_cats\hdf5\features.hdf5"
batch_size = 16
buffer_size = 1000
# %% ##################################################################
print(f"[INFO] loading images......")
image_paths = list(paths.list_images(dataset_path))
random.shuffle(image_paths)

# extract labels and then encode
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)
# %% ##################################################################
print(f"[INFO] loading network......")
model = ResNet50(weights="imagenet", include_top=False)

# initialise the dataset for extracted features
extracted_features_dataset = HDF5DatasetWriter(dims=(len(image_paths), 100352),
                                               output_path=extracted_features_output,
                                               buf_size=buffer_size)
extracted_features_dataset.store_class_labels(le.classes_)
# %% ##################################################################
# initialize the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths),
                               widgets=widgets).start()

# loop over the images in batches
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    batch_labels = labels[i: i + batch_size]

    batch_images = []
    for j, image_path in enumerate(batch_paths):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        batch_images.append(image)

    # pass the batched images thru the model
    batch_images = np.vstack(batch_images)
    features = model.predict(batch_images, batch_size=batch_size)  # type: np.ndarray
    features = features.reshape((features.shape[0], 100352))
    extracted_features_dataset.add(features, batch_labels)
    pbar.update(i)

extracted_features_dataset.close()
pbar.finish()
# %% ##################################################################
