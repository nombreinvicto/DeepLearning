from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.preprocessing import AspectAwarePreprocessor
from loader_util.io import HDF5DatasetWriter
from imutils import paths
from tqdm import tqdm
import numpy as np
import config
import json
import cv2
import os

# %% ##################################################################
train_paths = list(paths.list_images(config.images_path))
train_labels = [path.split(os.path.sep)[-1].split(".")[0] for path in train_paths]
le = LabelEncoder()
encoded_labels = le.fit_transform(train_labels)
print(le.classes_)
# %% ##################################################################
# do the datasplit
trainp, testp, trainy, testy = train_test_split(train_paths,
                                                encoded_labels,
                                                test_size=config.num_test_images,
                                                stratify=encoded_labels,
                                                random_state=42,
                                                shuffle=True)
trainp, valp, trainy, valy = train_test_split(trainp,
                                              trainy,
                                              test_size=config.num_val_images,
                                              stratify=trainy,
                                              random_state=42,
                                              shuffle=True)
datasets = [
    ("train", trainp, trainy, config.train_hdf5_path),
    ("test", testp, testy, config.test_hdf5_path),
    ("val", valp, valy, config.val_hdf5_path),
]

# %% ##################################################################
# preprocessors
aap = AspectAwarePreprocessor(256, 256)
r, g, b = ([], [], [])
# %% ##################################################################
for dtype, paths, labels, output_path in datasets:
    print(f"[INFO] building {dtype} dataset at : {output_path}......")
    writer = HDF5DatasetWriter(dims=(len(paths), 256, 256, 3),
                               outpath=output_path)
    writer.store_string_feature_labels(class_labels=le.classes_)
    with tqdm(total=len(paths), desc="processing images") as pbar:
        for path, label in zip(paths, labels):
            # remember this reads BGR
            image = cv2.imread(path)
            image = aap.preprocess(image)
            if dtype == "train":
                bmean, gmean, rmean = cv2.mean(image)[:3]
                r.append(rmean)
                g.append(gmean)
                b.append(bmean)
            writer.add([image], [label])
            pbar.update(1)
        writer.close()
# %% ##################################################################
# serialise the means
print(f"[INFO] serialising the means......")
D = {"r": np.mean(r), "g": np.mean(g), "b": np.mean(b)}
with open(config.dataset_mean_path, "w") as json_file:
    json.dump(D, json_file)
# %% ##################################################################
print(f"[INFO] completed building the hdf5 datasets......")