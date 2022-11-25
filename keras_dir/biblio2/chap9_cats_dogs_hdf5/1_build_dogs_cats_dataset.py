import dogs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.preprocessing import AspectAwarePreprocessor
from loader_util.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# %% ##################################################################
# grab the paths to the images in the dogs_cats_dataset
train_paths = list(paths.list_images(config.IMAGES_PATH))
train_labels = [path.split(os.path.sep)[-1].split(".")[0]
                for path in train_paths]
# %% ##################################################################
# encode the labels
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
# %% ##################################################################
# do the train test split using stratified sampling
train_paths, test_paths, train_labels, test_labels = train_test_split(train_paths,
                                                                      train_labels,
                                                                      test_size=config.NUM_TEST_IMAGES,
                                                                      random_state=42,
                                                                      stratify=train_labels)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths,
                                                                    train_labels,
                                                                    test_size=config.NUM_VAL_IMAGES,
                                                                    random_state=42,
                                                                    stratify=train_labels)

# %% ##################################################################

datasets = [
    ("train", train_paths, train_labels, config.TRAIN_HDF5_PATH),
    ("val", val_paths, val_labels, config.VAL_HDF5_PATH),
    ("test", test_paths, test_labels, config.TEST_HDF5_PATH)
]

# init the image processor
aap = AspectAwarePreprocessor(256, 256)
R, G, B = ([], [], [])
# %% ##################################################################
# loop over the dataset tuple
for (dtype, dpaths, dlabels, dout) in datasets:
    # create hdf5 db
    print(f"[INFO] building: {dout}......")
    writer = HDF5DatasetWriter(dims=(len(dpaths), 256, 256, 3),
                               output_path=dout)

    # init the progressbar
    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(dpaths),
                                   widgets=widgets).start()

    # loopover image paths
    for i, (dpath, dlabel) in enumerate(zip(dpaths, dlabels)):
        image = cv2.imread(dpath)
        image = aap.preprocess(image)

        if dtype == "train":
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add(rows=[image], labels=[dlabel])
        pbar.update(i)

    pbar.finish()
    writer.close()
# %% ##################################################################
# construct a dict of averages then srialise the means as json
print(f"[INFO] serialising means......")
D_means = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
with open(config.DATASET_MEAN_PATH, mode="w") as dataset_mean_file:
    dataset_mean_file.write(json.dumps(D_means))
# %% ##################################################################
print(f"[INFO] DONE with all datasets. EXITING script......")
