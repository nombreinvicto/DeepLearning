import os

IMAGES_PATH = r"C:\Users\mhasa\Downloads\dogs-vs-cats\train"

# total 25,000 images in dogs_vs_cats dataset
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define path to train, val and test hdf5s
BASE_PATH = r"C:\Users\mhasa\Downloads\dogs-vs-cats\hdf5"
TRAIN_HDF5_PATH = os.path.sep.join([BASE_PATH, "train.hdf5"])
VAL_HDF5_PATH = os.path.sep.join([BASE_PATH, "val.hdf5"])
TEST_HDF5_PATH = os.path.sep.join([BASE_PATH, "test.hdf5"])

OUTPUT_BASE_PATH = r"C:\Users\mhasa\Downloads\dogs-vs-cats\output"
MODEL_PATH = os.path.sep.join([OUTPUT_BASE_PATH, "alexnet_dog_vs_cat.model"])
DATASET_MEAN_PATH = os.path.sep.join([OUTPUT_BASE_PATH, "mean_dog_vs_cat.json"])
