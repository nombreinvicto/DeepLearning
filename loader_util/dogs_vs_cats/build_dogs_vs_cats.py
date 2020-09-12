# import the necessary packages

from loader_util.preprocessing import AspectAwarePreprocessor
from sklearn.model_selection import train_test_split
from config import dog_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from loader_util.io import HDF5DatasetWriter
from imutils import paths
from cv2 import cv2
import numpy as np
import progressbar
import json
import os

# %%

# grab the path to images
trainpaths = list(paths.list_images(config.IMAGES_PATH))
trainlabels = [p.split(os.path.sep)[-1].split(".")[0] for p in trainpaths]

# encode the labels to be later stored in the dataset
le = LabelEncoder()
trainlabels_encoded = le.fit_transform(trainlabels)

# %%

# perform stratified train test splits
all_train_paths, test_paths, all_trainy, testy = train_test_split(trainpaths,
                                                                  trainlabels_encoded,
                                                                  test_size=config.NUM_TEST_IMAGES,
                                                                  stratify=trainlabels,
                                                                  random_state=42)
train_paths, valid_paths, trainy, validy = train_test_split(all_train_paths,
                                                            all_trainy,
                                                            stratify=all_trainy,
                                                            random_state=42,
                                                            test_size=config.NUM_VAL_IMAGES)

# %%

# construct a structure to dataset flow
dataset_paths = [
    ("train", train_paths, trainy, config.TRAIN_HDF5),
    ("valid", valid_paths, validy, config.VALID_HDF5),
    ("test", test_paths, testy, config.TEST_HDF5)
]

# initialise the image preprocessor
aap = AspectAwarePreprocessor(256, 256)
r, g, b = [], [], []
# %%

# now initiate the actual preprocessing
for dtype, paths, labels, output in dataset_paths:
    # create the writer object
    print(f"Building Dataset of {dtype}.....")
    writer = HDF5DatasetWriter(dims=(len(paths), 256, 256, 3),
                               outputPath=output)

    # init the progressbar
    widgets = [f"Building Dataset Type: {dtype} ",
               progressbar.Percentage(),
               " ",
               progressbar.Bar(),
               " ",
               progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(paths)).start()

    # now loop over the image paths
    for i, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)  # type: np.ndarray
        image = aap.preprocess(image)

        if dtype == "train":
            b_, g_, r_ = cv2.mean(image)[:3]

            r.append(r_)
            g.append(g_)
            b.append(b_)

        writer.add([image], [label])
        pbar.update(i)

    # close the dataset after done with a certain split
    pbar.finish()
    writer.close()

# %%
# now store the RGB means to json file
print(f"Serialising the RGB means.....")
D = {"R": np.mean(r), "G": np.mean(g), "B": np.mean(b)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
# %%
