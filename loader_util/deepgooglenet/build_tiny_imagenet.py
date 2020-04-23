from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
from cv2 import cv2
import os

# %%

# grab paths to training images and then extract train class labels and
# encode them
trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perform stratified sampling form train set to construct test set
split = train_test_split(trainPaths, trainLabels,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=trainLabels, random_state=42)
trainpaths, testpaths, trainlabels, testlabels = split

# load the validation filename and then use mappings to build validaiton
# paths and labels lists
M = open(config.VAL_MAPPINGS).read().strip().split("\n")
M = [r.split("\t")[:2] for r in M]
valpaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.transform([m[1] for m in M])

# construct a list pairing the train, valid and test image paths along with
# their corresponding labels and output HDF5 files
datasets = [
    ('train', trainPaths, trainlabels, config.TRAIN_HDF5),
    ('val', valpaths, valLabels, config.VAL_HDF5),
    ('test', testpaths, testlabels, config.TEST_HDF5)
]

# initlaise the RGB averages
R, G, B = ([], [], [])

# create dataset - loop over the dataset tuples
for data_type, paths, labels, output in datasets:
    # create the HDF5 writer
    print(f"[INFO] building {output}.....")
    writer = HDF5DatasetWriter(dims=(len(paths), 64, 64, 3), outputPath=output)

    # init the progressbar
    widgets = [f"Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # now loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load image from disk
        image = cv2.imread(path)

        # if we are building the train set then compute the mean of each
        # channel in the image then update the respective lists
        if data_type == "train":
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            B.append(b)
            G.append(g)

        # add the image and label to HDF5
        writer.add([image], [label])
        pbar.update(i)

    # close the writer
    pbar.finish()
    writer.close()

# construct the dict of averages then serialise the means to a JSON file
print(f"[INFO] serialising means")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
