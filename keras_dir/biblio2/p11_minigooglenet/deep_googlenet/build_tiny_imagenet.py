import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar, json, cv2, os

# %%
train_paths = list(paths.list_images(config.train_images_path))
train_labels_wordnet = [pt.split(os.path.sep)[-3] for pt in train_paths]

le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels_wordnet)
# %%

# get the train and test data
split = train_test_split(train_paths,
                         train_labels_encoded,
                         test_size=config.num_test_images,
                         stratify=train_labels_encoded,
                         random_state=42)

train_paths, test_paths, train_labels, test_labels = split
# %%

# get the validation data
M = open(config.val_mapping).read().strip().split("\n")
M = [r.split("\t")[:2] for r in M]
val_paths = [os.path.join(config.valid_images_path, m[0]) for m in M]
val_labels = le.transform([m[1] for m in M])
# %%
# construct a list pairing the training, validation and testing image paths
datasets = [
    ("train", train_paths, train_labels, config.train_hdf5),
    ("val", val_paths, val_labels, config.valid_hdf5),
    ("test", test_paths, test_labels, config.test_hdf5)
]

R, G, B = [], [], []
# %%

# loop over the dataset tuples
for dtype, paths, labels, out_path in datasets:
    print(f"[INFO] building dataset: {out_path}......")
    writer = HDF5DatasetWriter(dims=(len(paths), 64, 64, 3),
                               outputPath=out_path)
    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
                                   widgets=widgets).start()

    # for a certain data type (train, test or valid) loop over the images to
    # create the dataset and add them to the database
    for i, (path, label) in enumerate(zip(paths, labels)):
        # load the image
        image = cv2.imread(path)

        if dtype == 'train':
            b, g, r = cv2.split(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add the image and label to dataset
        writer.add([image], [label])
        pbar.update(i)

    # close the writer
    pbar.finish()
    writer.close()
# %%

print(f"[INFO] serialising means......")
D = {
    "R": np.mean(R),
    "G": np.mean(G),
    "B": np.mean(B)
}

f = open(config.dataset_mean, mode="w")
f.write(json.dumps(D))
f.close()
#%%
