import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# %% ##################################################################
train_paths = list(paths.list_images(config.TRAIN_IMAGES))

# labels are the wnids from the path train\n01443537\images\n01443537_1.jpg
train_labels = [path.split(os.path.sep)[-3] for path in train_paths]

# encode the labels
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
# %% ##################################################################

train_paths, test_paths, train_labels, test_labels = train_test_split(train_paths,
                                                                      train_labels,
                                                                      random_state=42,
                                                                      test_size=config.NUM_TEST_IMAGES,
                                                                      stratify=train_labels)
# %% ##################################################################
M = open(config.VAL_MAPPING_FILE).read().strip().split("\n")

# M is now a list of lists, where each list contains
# validation image name and corresponding wnid
# e.g M = [[val_0.jpg, n123456]. [val_1.jpg, n45678].......]
M = [line.split("\t")[:2] for line in M]

# VAL_IMAGES = r"C:\Users\mhasa\Downloads\image-datasets\tiny-imagenet-200\val\images"
val_paths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
val_labels = le.transform([m[1] for m in M])
# %% ##################################################################

# cosntruct the list
datasets = [
    ('train', train_paths, train_labels, config.TRAIN_HDF5_FILEPATH),
    ('val', val_paths, val_labels, config.VAL_HDF5_FILEPATH),
    ('test', test_paths, test_labels, config.TEST_HDF5_FILEPATH),
]
# %% ##################################################################

# init the RGB list
R, G, B = ([], [], [])
# %% ##################################################################

# finally loop over the datasets tuples
for dtype, paths, labels, output_path in datasets:
    print(f"[INFO] building: {dtype} set......")
    writer = HDF5DatasetWriter(dims=(len(paths), 64, 64, 3),
                               output_path=output_path)

    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
                                   widgets=widgets).start()

    # loop over the images of the current dataset
    for i, (path, label) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        if dtype == 'train':
            b_mean, g_mean, r_mean = cv2.mean(image)[:3]
            R.append(r_mean)
            G.append(g_mean)
            B.append(b_mean)

        # add the image and label to the hdf5 dataset
        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()
# %% ##################################################################
# construct a dict of averages then serialise it
print(f"[INFO] serializing means......")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN_PATH, mode="w")
f.write(json.dumps(D))
f.close()
# %% ##################################################################

