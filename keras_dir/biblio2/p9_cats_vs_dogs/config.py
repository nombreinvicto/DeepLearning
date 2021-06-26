# define the paths to images directory
import os

IMAGES_PATH = r"C:\Users\mhasa\Google Drive\Tutorial " \
              r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\all_cats_dogs\train"

# DEFINE PATH TO TRAIN, VALID AND TEST HDF5S
HDF5_OUTPUT_DIR = R"C:\Users\mhasa\Google Drive\Tutorial " \
                  R"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\all_cats_dogs\hdf5"
MODEL_BASE_DIR = r"C:\Users\mhasa\Google Drive\Tutorial " \
                 r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\all_cats_dogs\model"
MISC_OUTPUT_DIR = r"C:\Users\mhasa\Google Drive\Tutorial " \
                  r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\all_cats_dogs\outputs"

NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

TRAIN_HDF5 = os.path.join(HDF5_OUTPUT_DIR, "train.hdf5")
VALID_HDF5 = os.path.join(HDF5_OUTPUT_DIR, "val.hdf5")
TEST_HDF5 = os.path.join(HDF5_OUTPUT_DIR, "test.hdf5")
MODEL_PATH = os.path.join(MODEL_BASE_DIR, "alexnet_dogs_vs_cats.model")

# path to dataset mean
DATASET_MEAN_PATH = os.path.join(MISC_OUTPUT_DIR,
                                 "alexnet_dogs_vs_cats_mean.json")
