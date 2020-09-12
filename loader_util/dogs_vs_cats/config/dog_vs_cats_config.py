# define the path to the images directory
IMAGES_PATH = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\all_cats_dogs"

NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define path to output hdf5 files
TRAIN_HDF5 = f"{IMAGES_PATH}//hdf5//train.hdf5"
VALID_HDF5 = f"{IMAGES_PATH}//hdf5//valid.hdf5"
TEST_HDF5 = f"{IMAGES_PATH}//hdf5//test.hdf5"

# define path to output of the model
MODEL_PATH = f"{IMAGES_PATH}//outputs//alexnet_dogs_cats.pt"

# define the path to the dataset mean for the mean preprocessor
DATASET_MEAN = f"{IMAGES_PATH}//outputs//dogs_cats_mean.json"
