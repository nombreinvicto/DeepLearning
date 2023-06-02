# import the necessary packages
from loader_util.preprocessing import AspectAwarePreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from loader_util.io import HDF5DatasetWriter
from collections import namedtuple
from imutils import paths
import numpy as np
import progressbar
import config
import json
import cv2
import os

# %% ##################################################################

# grab the paths to images
train_paths = list(paths.list_images(config.IMAGES_PATH))

# C:\Users\mhasa\Downloads\dogs-vs-cats\train\cat.100.jpg
train_labels = [pth.split(os.path.sep)[-1].split(".")[0]
                for pth in train_paths]
# encode labels
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
# %% ##################################################################

# separate the test_paths and test_labels
train_paths, test_paths, trainy, testy = train_test_split(train_paths,
                                                          train_labels_encoded,
                                                          test_size=config.NUM_TEST_IMAGES,
                                                          stratify=train_labels_encoded,
                                                          random_state=42)
# from the train_paths and trainy do another extraction to
# separate the validation_paths and validation labels
train_paths, val_paths, trainy, valy = train_test_split(train_paths,
                                                        trainy,
                                                        test_size=config.NUM_VAL_IMAGES,
                                                        stratify=trainy,
                                                        random_state=42)

# %% ##################################################################
dataset = namedtuple("dataset", "name paths labels hdf5_path")
datasets = [
    dataset("train", train_paths, trainy, config.TRAIN_HDF5_PATH),
    dataset("val", val_paths, valy, config.VAL_HDF5_PATH),
    dataset("test", test_paths, testy, config.TEST_HDF5_PATH)
]
# %% ##################################################################
# init the preprocessors
aap = AspectAwarePreprocessor(256, 256)
R, G, B = [], [], []

# %% ##################################################################
# start building the dataset
for dset in datasets:
    print(f"[INFO] building: {dset.hdf5_path}......")
    writer = HDF5DatasetWriter(dims=(len(dset.paths), 256, 256, 3),
                               outpath=dset.hdf5_path)

    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(dset.paths),
                                   widgets=widgets).start()

    # now loop over the paths
    for i, (path, label) in enumerate(zip(dset.paths, dset.labels)):
        # load the image and process it
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if dset.name == "train":
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add preprocessed image to the hdf5 writer buffer
        writer.add([image], [label])
        pbar.update(i)

    # close the db for the current split = train/val/test
    pbar.finish()
    writer.close()
# %% ##################################################################
# after done with ALL splits, serialise the mean of means
print(f"[INFO] serialising means......")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
with open(config.DATASET_MEAN_PATH, mode="w") as file_pointer:
    file_pointer.write(json.dumps(D))
# %% ##################################################################
