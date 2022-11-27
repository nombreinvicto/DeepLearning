import os

# define the paths to the images directory
IMAGES_PATH = r"C:\Users\mhasa\Downloads\delete\train\train"

# train/test data
# total 25_000 images in the dataset
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# hdf5 output paths
project_dir = r"C:\Users\mhasa\PycharmProjects\deep_learning"
hdf5_dir = os.path.sep.join([project_dir,
                             "loader_util",
                             "datasets",
                             "kaggle_dogs_cats",
                             "hdf5"])
os.makedirs(hdf5_dir, exist_ok=True)

# set output save paths
TRAIN_HDF5_PATH = os.path.sep.join([hdf5_dir, "train.hdf5"])
VAL_HDF5_PATH = os.path.sep.join([hdf5_dir, "val.hdf5"])
TEST_HDF5_PATH = os.path.sep.join([hdf5_dir, "test.hdf5"])

# output dir
OUTPUT_DIR = os.path.sep.join([project_dir,
                               "loader_util",
                               "datasets",
                               "kaggle_dogs_cats",
                               "output"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.sep.join([OUTPUT_DIR, "alexnet_dogs_cats.model"])
DATASET_MEAN_PATH = os.path.sep.join([OUTPUT_DIR, "dogs_cats_mean.json"])
