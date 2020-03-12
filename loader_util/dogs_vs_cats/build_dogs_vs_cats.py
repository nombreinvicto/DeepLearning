# import the necessary packages
from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.preprocessing import AspectAwarePreprocessor
from loader_util.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
from cv2 import cv2
import os

# %%

# grab the paths to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[-1].split('.')[0] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perform stratified sampling frm the trainig set to build test split from
# train data
split = train_test_split(trainPaths,
                         trainLabels,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=trainLabels,
                         random_state=42)
trainPaths, testPaths, trainLabels, testLabels = split

# perform another stratified sampling this time to build the validation data
split = train_test_split(trainPaths,
                         trainLabels,
                         test_size=config.NUM_VAL_IMAGES,
                         stratify=trainLabels,
                         random_state=42)
trainPaths, valPaths, trainLabels, valLabels = split

# construct a list pairing the training, validation and testing image paths
# along with their corresponding labels and output HDF5 files
datasets = [
    ('train', trainPaths, trainLabels, config.TRAIN_HDF5),
    ('val', valPaths, valLabels, config.VALID_HDF5),
    ('test', testPaths, testLabels, config.TEST_HDF5)
]

# initialise the image preprocessor and the lists of RGB channel averages
aap = AspectAwarePreprocessor(256, 256)
R, G, B = ([], [], [])

# loop over the dataset tuples
for dtype, paths, labels, outpath in datasets:
    # create the HDF5 writer
    print(f'[INFO] building {outpath}')
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath=outpath)

    # initialise the progrssbar
    widgets = ['Building Dataset: ', progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # now loop over the iamge paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and process it
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # if we are building train dataset then compute mean of each channel
        if dtype == 'train':
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

         # add the image and label to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

    # close the HDF5 writer
    pbar.finish()
    writer.close()

# construct the dictionary of averages then serialise the means to JSON
print(f'[INFO] serialising means')
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()

