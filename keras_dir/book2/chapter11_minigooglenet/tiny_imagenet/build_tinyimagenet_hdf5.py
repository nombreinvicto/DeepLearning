# import the necessary packages
from loader_util.projects.tiny_imagenet import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.io import HDF5DatasetWriter
from collections import namedtuple
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# %% ##################################################################
dataset = namedtuple("dataset", "type paths labels output_path")
# %% ##################################################################
train_paths = list(paths.list_images(config.train_images_path))

# r"C:\Users\mhasa\Downloads\tiny-imagenet-200\train\n1234\images\n1234_0.jpg"
train_wnid_labels = [pth.split(os.path.sep)[-3] for pth in train_paths]

# encode the labels
le = LabelEncoder()
trainlabels_enc = le.fit_transform(train_wnid_labels)
# %% ##################################################################
train_paths, test_paths, train_labels, test_labels = train_test_split(train_paths,
                                                                      trainlabels_enc,
                                                                      test_size=config.num_test_images,
                                                                      random_state=42,
                                                                      stratify=trainlabels_enc)

# %% ##################################################################
# load the validation filename to wnid mapping file

# ["val_0.jpeg n1234", "val_1.jpeg n2345".....]
M = open(config.val_mappings_file).read().strip().split("\n")

# [["val_0.jpeg", "n1234"], ["val_1.jpeg", "n2345"], .....]
M = [r.split("\t")[:2] for r in M]

val_paths = [os.path.sep.join([config.val_images_path, m[0]]) for m in M]
val_labels = le.transform([m[1] for m in M])
# %% ##################################################################
datasets = [
    dataset(type="train", paths=train_paths, labels=train_labels, output_path=config.train_hdf5_path),
    dataset(type="val", paths=val_paths, labels=val_labels, output_path=config.val_hdf5_path),
    dataset(type="test", paths=test_paths, labels=test_labels, output_path=config.test_hdf5_path)
]

# %% ##################################################################
# init the list of RGB channel averages
R, G, B = [], [], []
# %% ##################################################################
print(f"[INFO] started creating dataset......")
for dataset in datasets:
    print(f"[INFO] building {dataset.type} set of size {len(dataset.paths)} at {dataset.output_path}......")
    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(dataset.paths),
                                   widgets=widgets).start()

    # init the hdf5 writer
    writer = HDF5DatasetWriter(dims=(len(dataset.paths), 64, 64, 3),
                               outpath=dataset.output_path)

    # loop over the image paths of the current dataset
    for i, (path, label) in enumerate(zip(dataset.paths, dataset.labels)):
        image = cv2.imread(path)
        if dataset.type == "train":
            b, g, r = cv2.split(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])
        pbar.update(i)

    # close connections when done with a particular dataset
    pbar.finish()
    writer.close()

print(f"[INFO] done with all datasets......")
# %% ##################################################################
print(f"[INFO] serialising means......")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.rgb_mean_path, "w")
f.write(json.dumps(D))
f.close()
# %% ##################################################################
