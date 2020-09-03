# import the necessary packages
import numpy as np
import seaborn as sns
sns.set()
# %%
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from loader_util.io import HDF5DatasetWriter
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from imutils import paths
import random
import progressbar
import os

# %%

# construct the argument parser
args_dict = {
    'dataset': r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\flowers17\images",
    'output': r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\flowers17\extracted_features\features.hdf5",
    "batch_size": 32,
    "buffer_size": 1000
}

out_shape = 512 * 7 * 7

# %%

# grab list of images and then random shuffle
print(f"[INFO] loading images.....")
image_paths = list(paths.list_images(args_dict['dataset']))
random.shuffle(image_paths)

# extract class labels and them encode them
labels = [p.split(os.path.sep)[-2] for p in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)
# %%

# load the vgg16 net
print(f"[INFO] loading network .....")
model = VGG16(weights="imagenet", include_top=False)

# initialise the HDF5 dataset
dataset = HDF5DatasetWriter(dims=(len(image_paths), out_shape),
                            outputPath=args_dict['output'],
                            dataKey="extracted_features",
                            bufSize=args_dict['buffer_size'])


dataset.storeClassLabels(le.classes_)
#%%

# initialise the progressbar
widgets = ["Extracting Features:   ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

# lopp over the images in batches
for i in range(0, len(image_paths), args_dict['batch_size']):
    batched_paths = image_paths[i:i+args_dict['batch_size']]
    batched_labels = labels[i:i+args_dict['batch_size']]
    batched_images = []

    # loop over the images and labels in the current batch
    for j, image_path in enumerate(batched_paths):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # needs change here for custom datasets
        image = imagenet_utils.preprocess_input(image)

        # add the image to batch
        batched_images.append(image)

    # pass the images thru the net
    batched_images = np.vstack(batched_images)

    extracted_features = model.predict(batched_images,
                                       batch_size=args_dict['batch_size'])

    reshaped_features = extracted_features.\
        reshape((extracted_features.shape[0], out_shape))

    # add the features and labels to hdf5 dataset
    dataset.add(reshaped_features, batched_labels)
    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()




































