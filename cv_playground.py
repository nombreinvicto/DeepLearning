# gp = 14(4.333 x 3) + 10(4 x 3) + (3.667 x 3) + (3.333 x 3)
sofar = 18_80_868.89, 21_030.77


#https://www.linkedin.com/in/mhasan3/

#https://www.mhasan3.com/




from tensorflow.keras.layers import Conv2D, MaxPooling2D

m = Conv2D()

# import the necessary packages
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loader_util.preprocessing import ImageToArrayPreprocessor, \
    AspectAwarePreprocessor, SimplePreProcessor, MeanPreprocessor
from loader_util.io import HDF5DatasetGenerator
from loader_util.datasets import SimpleDatasetLoader
from loader_util.nn.conv import FCHeadNet
from loader_util.callbacks import EpochCheckpoint, TrainingMonitor
##
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model





