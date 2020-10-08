
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from loader_util.io import HDF5DatasetWriter
from loader_util.preprocessing import AspectAwarePreprocessor, \
    MeanSubtractionPreProcessor
from imutils import paths
import numpy as np
import progressbar
from cv2 import cv2
import os

# %%

# imagePath = r"C:\Users\mhasa\Google Drive\Tutorial
# Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\Unique3DClusters"
imagePath = r"dataset _mvcnn_color_roi_retrieve_model_28px1px_255.hdf5"
dbPath = r"C:\Users\mhasa\Desktop"

# grab paths to training images and then extract train class labels and encode
trainPaths = list(paths.list_images(imagePath))

trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]
print("Unique Classes: ", len(np.unique(trainLabels)))
# %%
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
# %%
class_labels = np.array(le.classes_)
# %%

# perform stratified sampling from train set to construct validation set
split = train_test_split(trainPaths,
                         trainLabels,
                         test_size=0.3,
                         stratify=trainLabels,
                         random_state=42)
trainpaths, testpaths, trainlabels, testlabels = split


# %%

# construct list pair
datasets = [
    ('train', trainpaths, trainlabels,
     f"{dbPath}//train_mvcnn_color_roi_10class_32px1px_255.hdf5"),
    ('val', testpaths, testlabels,
     f"{dbPath}//test_mvcnn_color_roi_10class_32px1px_255.hdf5")
]
# %%

## Global Variables
IMAGE_READ_MODE = [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE]

TARGET_SIZE = 32
CHANNLE_DIM = 1
IMAGE_READ_INDEX = 1

# initialise the preprocessors
aap = AspectAwarePreprocessor(TARGET_SIZE, TARGET_SIZE)
mp = MeanSubtractionPreProcessor()

# create dataset loop over the dataset tuples
for dataType, paths, labels, output in datasets:
    # create HDF5 writer
    print(f"[INFO] building {output}.....")
    writer = HDF5DatasetWriter(dims=(len(paths),
                                     TARGET_SIZE,
                                     TARGET_SIZE,
                                     CHANNLE_DIM),
                               outputPath=output)

    # initialise the progressbar
    widgets = [f"Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # now loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and preprocess it
        image = cv2.imread(path, IMAGE_READ_MODE[IMAGE_READ_INDEX])
        image = aap.preprocess(image)
        #image = mp.preprocess(image)
        image = image.astype('float32')
        image = np.expand_dims(image, axis=-1)

        image = image / 255.0 # dont use for gan sets

        # add the image and label to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

    # store class labels before exiting
    writer.storeClassLabels(class_labels)

    # close the writer
    pbar.finish()
    writer.close()
# %%
