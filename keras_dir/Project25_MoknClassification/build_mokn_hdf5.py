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
#%%

#imagePath = r"C:\Users\mhasa\Google Drive\Tutorial
# Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\Unique3DClusters"
imagePath = r"C:\Users\mhasa\Desktop\MOKN_6Cluster"

# grab paths to training images and then extract train class labels and encode
trainPaths = list(paths.list_images(imagePath))

# for class
n_class = 15
#trainPaths = trainPaths[0: n_class]

trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]
print("Unique Classes: ", len(np.unique(trainLabels)))
#%%
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
#%%

# perform stratified sampling from train set to construct validation set
split = train_test_split(trainPaths,
                         trainLabels,
                         test_size=0.3,
                         stratify=trainLabels,
                         random_state=42)
trainpaths, testpaths, trainLabels, testLabels = split
#%%

# construct list pair
datasets = [
    ('train', trainpaths, trainLabels,
     f"{imagePath}//train_mokn_6class.hdf5"),
    ('val', testpaths, testLabels,
     f"{imagePath}//validate_mokn_6class.hdf5")
]
#%%

# initialise the preprocessors
aap = AspectAwarePreprocessor(224, 224)
mp = MeanSubtractionPreProcessor()

# create dataset loop over the dataset tuples
for dataType, paths, labels, output in datasets:
    # create HDF5 writer
    print(f"[INFO] building {output}.....")
    writer = HDF5DatasetWriter(dims=(len(paths), 224, 224, 3), outputPath=output)

    # initialise the progressbar
    widgets = [f"Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # now loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and preprocess it
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = aap.preprocess(image)
        image = mp.preprocess(image)

        # add the image and label to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

    # close the writer
    pbar.finish()
    writer.close()

