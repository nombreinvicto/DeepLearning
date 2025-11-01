import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes
sns.set()
#%%
# import the necessary keras packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loader_util.preprocessing import ImageToArrayPreprocessor, AspectAwarePreprocessor
from loader_util.datasets import SimpleDatasetLoader
from loader_util.nn.conv import FCHeadNet
##
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from imutils import paths
#%%

data_dir = r"C:\Users\mhasa\Desktop\mvcnn"
dest_dir = r"C:\Users\mhasa\Desktop\mvcnn_reorg"
path_list = list(paths.list_images(data_dir))
#%%
# now start the move
for path in path_list:
    category_folder = path.split(os.path.sep)[-4]
    shutil.copy(src=path,
                dst=f"{dest_dir}//{category_folder}")
#%%
