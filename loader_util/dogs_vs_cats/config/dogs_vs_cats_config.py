# define the paths to the images directory
import os

BASE = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\DeepLearningCV\loader_util\datasets\kaggle_dogs_vs_cats"
OUTPUT_BASE = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\DeepLearningCV\loader_util\dogs_vs_cats\output"
IMAGES_PATH = os.path.join(BASE, 'train')

NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

# define the path to the ouput train, valid and test HDF5 files
r"../datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5"
TRAIN_HDF5 = os.path.join(BASE, 'hdf5', 'train.hdf5')
r"../datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5"
VALID_HDF5 = os.path.join(BASE, 'hdf5', 'val.hdf5')
r"../datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5"
TEST_HDF5 = os.path.join(BASE, 'hdf5', 'test.hdf5')

# path to the output model file
"output/alexnet_dogs_vs_cats.model"
MODEL_PATH = os.path.join(OUTPUT_BASE, 'alexnet_dogs_vs_cats.model')

# define the path to the dataset mean
r"output/dogs_vs_cats_mean.json"
DATASET_MEAN = os.path.join(OUTPUT_BASE, 'dogs_vs_cats_mean.json')

# define the path to general purpose output directory
OUTPUT_PATH = OUTPUT_BASE
