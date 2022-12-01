# import the necessary packes
import os

# define the paths to the training and validation directories
# TRAIN_IMAGES dir contains images under id folders e.g -> n01443537\images\n01443537_1.jpg
TRAIN_IMAGES = r"C:\Users\mhasa\Downloads\image-datasets\tiny-imagenet-200\train"
VAL_IMAGES = r"C:\Users\mhasa\Downloads\image-datasets\tiny-imagenet-200\val\images"

# define the path to the file that maps val filenames to labels
# contains lines like -> val_0.JPEG	n03444034
VAL_MAPPING_FILE = r"C:\Users\mhasa\Downloads\image-datasets\tiny-imagenet-200\val\val_annotations.txt"

# define the paths to wordnet files needed to geenerate class labels
# this file is just a text file listing all 200 wnids
WORDNET_IDS_FILE = r"C:\Users\mhasa\Downloads\image-datasets\tiny-imagenet-200\wnids.txt"

# this file is nid tab separated by class name e.g n00001740 cell
WORD_LABELS_FILE = r"C:\Users\mhasa\Downloads\image-datasets\tiny-imagenet-200\words.txt"

# TRAIN/TEST params
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

TRAIN_HDF5_FILEPATH = r"C:\Users\mhasa\PycharmProjects\deep_learning\loader_util\datasets\tiny_imagenet\hdf5\train.hdf5"
VAL_HDF5_FILEPATH = r"C:\Users\mhasa\PycharmProjects\deep_learning\loader_util\datasets\tiny_imagenet\hdf5\val.hdf5"
TEST_HDF5_FILEPATH = r"C:\Users\mhasa\PycharmProjects\deep_learning\loader_util\datasets\tiny_imagenet\hdf5\test.hdf5"
DATASET_MEAN_PATH = r"C:\Users\mhasa\PycharmProjects\deep_learning\loader_util\datasets\tiny_imagenet\output\tiny_imagenet_mean.json"



