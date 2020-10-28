# %%

import os
import shutil
import random
from cv2 import cv2
import h5py
import numpy as np
import seaborn as sns
from imutils import paths

sns.set()

# %%

# import the necessary keras/sklearn packages

from sklearn.neighbors import KDTree
from tensorflow.keras.models import Model, load_model


# %%
def load_extracted_features(sig_file_name, normalise=False):
    # load the hdf5 dataset
    data_dir = r"C:\Users\mhasa\GDrive\mvcnn"
    signature_db = h5py.File(name=f"{data_dir}//{sig_file_name}",
                             mode="r")
    extracted_features = signature_db['extracted_features'][:]

    if normalise:
        # now normalise the feature matrix
        norm_matrix = np.linalg.norm(extracted_features, axis=1)
        norm_matrix = norm_matrix.reshape(norm_matrix.shape[0], 1)
        normalised_extracted_features = extracted_features / norm_matrix
        extracted_features = normalised_extracted_features

        sample_row_vector = normalised_extracted_features[0, :]
        magnitude_of_sample_vector = np.linalg.norm(sample_row_vector)
        print(f"Magnitude of sample vector: {magnitude_of_sample_vector}")

    print(f"Shape of extracted Features: {extracted_features.shape}")
    return extracted_features, signature_db


# %%
def load_trained_cnn(model_name, layer_name):
    data_dir = r"C:\Users\mhasa\GDrive\mvcnn"
    layer_info = {
        'flatten_3136': 16,
        'dense_512': 17
    }

    # load the model
    loaded_model = load_model(f"{data_dir}//{model_name}")
    flatten_layer = loaded_model.layers[layer_info[layer_name]]
    retrieval_model = Model(inputs=loaded_model.input,
                            outputs=flatten_layer.output)
    print(retrieval_model.summary())
    return retrieval_model


# %%

# read the image and create signature
def create_image_signature(image_name, retrieval_model, normalise=True):
    data_dir = r"C:\Users\mhasa\GDrive\mvcnn"
    img = cv2.imread(f"{data_dir}//{image_name}.png", cv2.IMREAD_GRAYSCALE)
    img = img.astype('float32')
    img = img / 255.0

    # channel dim and batch dim since we doing feature extraction
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    signature = retrieval_model.predict(img)

    if normalise:
        signature = signature / np.linalg.norm(signature)

    return signature


# %%

def fit_kdtree(feature_matrix):
    kdtree = KDTree(feature_matrix)
    return kdtree


def query_kdtree(kdtree, sample_signature, k=5):
    # distances are returned sorted, not indices
    dist, ind = kdtree.query(sample_signature, k=k)
    return dist[0], ind[0]


# %%

# load the extracted features matrix and then normalise
features, signature_db = load_extracted_features \
    (sig_file_name="data_sig_col_roi_28px_255_minvgg.hdf5", normalise=True)

# load the trained CNN model
cnn_model = load_trained_cnn \
    (model_name="model_mvcnn_color_roi_10class_28px1px_255_minvgg.h5",
     layer_name="flatten_3136")

# now get closes distances and indices from kdtree - dists are sorted
kdtree = fit_kdtree(features)
# %%
# create sample image signature
sample_signature = create_image_signature("sample_gear",
                                          cnn_model,
                                          normalise=True)

# distances are always returned sorted, not the indices
dist, positions = query_kdtree(kdtree, sample_signature)
# %%

# get he neighbor names
# neighbor names are acc to sorted indices, these are unique names
indices_that_sort_positions = np.argsort(positions)
sorted_positions = np.sort(positions)
argsorted_dist = dist[indices_that_sort_positions]
neighbor_names = signature_db['label_names'][sorted_positions]
neighbor_names = np.array([n.decode() for n in neighbor_names])
labels = signature_db['labels'][sorted_positions]

# %%

target_dir = r"C:\Users\mhasa\Desktop\neighbors"

pristine_retrieve_models_dir = r"C:\Users\mhasa\Desktop\retrieval_models"
image_paths = list(paths.list_images(pristine_retrieve_models_dir))
# %%

for path in image_paths:
    image_name = path.split(os.path.sep)[-1]

    if image_name in neighbor_names:
        position_of_neighbor = np.where(neighbor_names == image_name)[0][0]
        euclid_dist = str(argsorted_dist[position_of_neighbor])[:5]
        assigned_label = labels[position_of_neighbor]
        dst = f"{target_dir}//D={euclid_dist}__L={assigned_label}__{image_name}"
        shutil.copy2(src=path, dst=f"{dst}")
# %%