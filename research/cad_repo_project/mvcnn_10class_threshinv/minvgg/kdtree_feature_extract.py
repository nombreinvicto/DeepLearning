# %%

import os
import random
import json
import shutil
from cv2 import cv2
import h5py
import numpy as np
from imutils import paths
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
    # data_dir = r"C:\Users\mhasa\GDrive\mvcnn"
    # img = cv2.imread(f"{data_dir}//{image_name}.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
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


def query_kdtree(kdtree, sample_signature, k=10):
    # distances are returned sorted, not indices
    dist, ind = kdtree.query(sample_signature, k=k)
    return dist[0], ind[0]


# %%
print(f'[INFO] loading signature matrix and trained CNN.....')
# load the extracted features matrix and then normalise
features, signature_db = load_extracted_features \
    (sig_file_name="data_sig_col_roi_28px_255_minvgg.hdf5", normalise=True)

# load the trained CNN model
cnn_model = load_trained_cnn \
    (model_name="model_mvcnn_color_roi_10class_28px1px_255_minvgg.h5",
     layer_name="flatten_3136")

print(f'[INFO] fitting KDTREE.....')
# now get closes distances and indices from kdtree - dists are sorted
kdtree = fit_kdtree(features)

# %%
## this is where we start all the hanky panky stuff
print(f'[INFO] initialising variables and directories for MAP '
      f'calculation.....')
queries = 1000
current_query = 0

# these paths for the p@K and AP values storage files
result_dir = r"C:\Users\mhasa\Desktop\map_out"
ap_file = f"{result_dir}//avgPrecision.txt"
pk_json = f"{result_dir}//p_at_k.json"

mvcnn_query_image_dir = r"C:\Users\mhasa\Desktop\mvcnn_gray_roi_28px"
mvcnn_query_pristine_dir = r"C:\Users\mhasa\Desktop\mvcnn_reorg"
neigh_target_dir = r"C:\Users\mhasa\Desktop\neighbors"
pristine_retrieve_models_dir = r"C:\Users\mhasa\Desktop\retrieval_models"
query_part_visualisation_dir = r"C:\Users\mhasa\Desktop\query_part"

pristine_retrieve_image_paths = list(
    paths.list_images(pristine_retrieve_models_dir))
mvcnn_image_paths = list(paths.list_images(mvcnn_query_image_dir))
unique_cats = np.unique([p.split(os.path.sep)[-2] for p in mvcnn_image_paths])
mvcnn_image_paths = np.array(mvcnn_image_paths)  # type: np.ndarray

# initialise the json file
init_dict = {}
for cat in unique_cats:
    init_dict[cat] = []

# now write this init dict in the json file
with open(pk_json, mode="w") as json_file:
    json_file.write(json.dumps(init_dict))
# %%
print(f'[INFO] Entering loop.....')
## now start the actual routine
while current_query <= queries:
    np.random.shuffle(
        mvcnn_image_paths)  # these are the 28px gray mvccn images
    random_chosen_path = random.choice(mvcnn_image_paths)
    random_chosen_cat = random_chosen_path.split(os.path.sep)[-2]
    random_chosen_name = random_chosen_path.split(os.path.sep)[-1]

    # create sample image signature
    sample_signature = create_image_signature(random_chosen_path,
                                              cnn_model,
                                              normalise=True)

    # distances are always returned sorted, not the indices
    dist, positions = query_kdtree(kdtree, sample_signature)

    # neighbor names are acc to sorted indices, these are unique names
    indices_that_sort_positions = np.argsort(positions)
    sorted_positions = np.sort(positions)
    argsorted_dist = dist[indices_that_sort_positions]
    neighbor_names = signature_db['label_names'][sorted_positions]
    neighbor_names = np.array([n.decode() for n in neighbor_names])
    labels = signature_db['labels'][sorted_positions]

    # now transfer the pristine retrieval models to target for visual observati
    # this is the ACTUAL retrieve model dir
    for path in pristine_retrieve_image_paths:
        image_name = path.split(os.path.sep)[-1]

        if image_name in neighbor_names:
            position_of_neighbor = np.where(neighbor_names == image_name)[0][0]
            euclid_dist = str(argsorted_dist[position_of_neighbor])[:5]
            assigned_label = labels[position_of_neighbor]
            dst = f"{neigh_target_dir}//D={euclid_dist}__L={assigned_label}" \
                  f"__{image_name}"
            shutil.copy2(src=path, dst=f"{dst}")

    # also copy the ACTUAL mvcnn reference image i.e query image
    shutil.copy2(src=f"{mvcnn_query_pristine_dir}//{random_chosen_cat}//"
                     f"{random_chosen_name}",
                 dst=f"{query_part_visualisation_dir}//{current_query + 1}__"
                     f"{random_chosen_name}")

    # as you observe wait for filter input
    relevance_filter = input(
        f"Enter Filter for query part {random_chosen_cat}:  ")
    if relevance_filter == "q":
        print("Gracefully Exiting .....")
        break

    # once relevance filter supplied, delete the images
    images_to_delete = os.listdir(neigh_target_dir)
    for image_file in images_to_delete:
        os.remove(f"{neigh_target_dir}//{image_file}")
    print("Deletion Done .....")

    # now purge p@K values and AP value to files
    filter_nums = np.asarray([int(char) for char in relevance_filter])
    ground_truth_positives = np.sum(filter_nums)
    p_at_k_values = []
    k = len(filter_nums)

    # calculate p at k values for this current query
    for i in range(k):
        docs_to_consider = filter_nums[0:i + 1]
        denom = len(docs_to_consider)
        numer = np.sum(docs_to_consider)
        p_at_k_values.append(round(numer / denom, 3))

    # calculate the AP value for this current query
    AP = 0
    for p_at_k, relevance in zip(p_at_k_values, filter_nums):
        AP += p_at_k * relevance
    AP = round(AP / (ground_truth_positives + 1e-5), 3)

    # now purge the values - first the AP value in the text file
    with open(ap_file, mode="a") as apfile:
        apfile.write(str(AP) + "\n")

    with open(pk_json, mode="r+") as jsonfile:
        stored_dict = json.loads(jsonfile.read())
        jsonfile.seek(0)
        jsonfile.truncate()
        stored_dict[str(random_chosen_cat)].append(p_at_k_values)
        jsonfile.write(json.dumps(stored_dict))
    print(f"Purging done for part: {random_chosen_cat}.....")

    current_query += 1
# %%
