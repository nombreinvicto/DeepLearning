# import the required packages
from os import path

# define the paths to training and validation directories
TRAIN_IMAGES = r"/home/mhasan3/Desktop/WorkFolder/tiny-imagenet-200/train/"
VAL_IMAGES = r"/home/mhasan3/Desktop/WorkFolder/tiny-imagenet-200/val/images/"

# define the path to file that maps validation filenames to corr class labels
VAL_MAPPINGS = r"/home/mhasan3/Desktop/WorkFolder/tiny-imagenet-200/val" \
               r"/val_annotations.txt"

# define paths to WordNet hierarchy files used to generate class labels
WORDNET_IDS = r"/home/mhasan3/Desktop/WorkFolder/tiny-imagenet-200/wnids.txt"
WORD_LABELS = r"home/mhasan3/Desktop/WorkFolder/tiny-imagenet-200/words.txt"

# since we do not have access to testing data we need to take number of
# images from the training data and use it to instead
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to output train, valid and test HDF5 files
TRAIN_HDF5 = r"/home/mhasan3/Desktop/WorkFolder/tiny_imagenet_data/hdf5" \
             r"/train.hdf5"
VAL_HDF5 = r"/home/mhasan3/Desktop/WorkFolder/tiny_imagenet_data/hdf5" \
           r"/val.hdf5"
TEST_HDF5 = r"/home/mhasan3/Desktop/WorkFolder/tiny_imagenet_data/hdf5" \
            r"/test.hdf5"

# define the path to the dataset mean
DATASET_MEAN = r"output/tiny-image-net-200-mean.json"

# define the path to the output directory used for storing plots,
# classification reports
OUTPUT_PATH = "output"
MODEL_PATH = path.sep.join([OUTPUT_PATH, "checkpoints", "epoch_70.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH, "deepergooglenet_imagenet.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH, "deepergooglenet_imagenet.json"])


