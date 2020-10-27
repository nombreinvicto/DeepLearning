import os
from cv2 import cv2
import numpy as np
import pandas as pd
import progressbar
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes

sns.set()
# %%
# import the necessary keras packages
from imutils import paths
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from loader_util.io import HDF5DatasetWriter
##
from tensorflow.keras.models import Model, load_model


# %%

data_dir = r"C:\Users\mhasa\Desktop\retrieval_model_color_roi_28px"
model_dir = r"C:\Users\mhasa\GDrive\mvcnn"
batch_size = 32
# %%
print(f'[INFO] loading images.....')
image_paths = list(paths.list_images(data_dir))
random.shuffle(image_paths)

# extract the image names
image_names = [p.split(os.path.sep)[-1] for p in image_paths]
labels = [p.split(os.path.sep)[-2] for p in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)
#%%
print(f'[INFO] loading pretrained model.....')
loaded_model = load_model(f"{model_dir}//model_mvcnn_color_roi_10class_28px1px_255_minvgg.h5")
layer_info = {
    'flatten_3136': 16,
    'dense_512': 17
}

last_layer = loaded_model.layers[layer_info['dense_512']]
retrieval_model = Model(inputs=loaded_model.input, outputs=last_layer.output)
print(retrieval_model.summary())
vector_shape = last_layer.output.shape[1]
#%%
# init the hdf5 store for the vectors
dataset = HDF5DatasetWriter(dims=(len(image_paths), vector_shape),
                            outputPath=f"{model_dir}//data_sig_col_roi_28px_255_dense_last.hdf5",
                            dataKey="extracted_features",
                            bufSize=32)
dataset.storeClassLabels(image_names)
#%%
# init the progressbar
widgets = ["Extracting Features : ", " ",  progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

# loop over images in batches
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]
    batch_images = []

    # loop over images in the current batch
    for j, image_path in enumerate(batch_paths):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype('float32')
        img = img / 255.0

        # channel dim and batch dim since we doing feature extraction
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # add image to batch
        batch_images.append(img)

    # pass batch of images thru netowrk
    batch_images = np.vstack(batch_images)
    extracted_features = retrieval_model.predict(batch_images)
    dataset.add(extracted_features, batch_labels)
    pbar.update(i)

# close everything
dataset.close()
pbar.finish()
#%%
