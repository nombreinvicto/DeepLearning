import os
import numpy as np
import seaborn as sns
import progressbar

sns.set()
# %%
# import the necessary packages
import config, json, cv2
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.preprocessing import ImageToArrayPreprocessor, \
    AspectAwarePreprocessor
from loader_util.io import HDF5DatasetWriter
from imutils import paths

# %%

# grab the paths to the images
train_paths = list(paths.list_images(config.IMAGES_PATH))
train_labels = [pt.split(os.path.sep)[-1].split(".")[0] for pt in train_paths]

# encode the labels
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
# %%

# perform stratified sample
split = train_test_split(train_paths, train_labels,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=train_labels, random_state=42)
train_paths, test_paths, train_labels, test_labels = split

# now from within train paths get validation set paths
split = train_test_split(train_paths, train_labels,
                         test_size=config.NUM_VAL_IMAGES,
                         stratify=train_labels, random_state=42)
train_paths, val_paths, train_labels, val_labels = split
# %%

# construct a list pairing the datasets
datasets = [
    ("train", train_paths, train_labels, config.TRAIN_HDF5),
    ("val", val_paths, val_labels, config.VALID_HDF5),
    ("test", test_paths, test_labels, config.TEST_HDF5)
]

# initialise the image preprocessor
aap = AspectAwarePreprocessor(256, 256)
R, G, B = [], [], []
# %%
# finally build the dataset
for dtype, paths, labels, outpath in datasets:
    print(f"[INFO] building : {outpath}......")
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath=outpath)

    # init the progressbar
    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
                                   widgets=widgets).start()

    # loop over the image paths
    for i, (path, label) in enumerate(zip(paths, labels)):
        # load the image and preprocess
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if dtype == 'train':
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])
        print(dtype, i)
        pbar.update(i)

    # close the write
    pbar.finish()
    writer.close()
# %%

print(f"[INFO] serializing the means......")
D = {
    "R": np.mean(R),
    "G": np.mean(G),
    "B": np.mean(B),
}

with open(config.DATASET_MEAN_PATH, "w") as mean_file:
    mean_file.write(json.dumps(D))
# %%
