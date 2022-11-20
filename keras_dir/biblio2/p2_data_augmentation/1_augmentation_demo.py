import logging
import sys, os, json
import numpy as np
import pandas as pd
import seaborn as sns
import argparse, progressbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

sns.set()
# %% ##################################################################
logging_name = "augmentation_demo"
logger = logging.getLogger(logging_name)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s "
                              "%(levelname)s: "
                              "%(name)s: -> %(message)s",
                              "%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# %% ##################################################################
# import the necessary packages
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% ##################################################################
# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path_to_input_image")
ap.add_argument("-o", "--output", required=True, help="output_path")
ap.add_argument("-p", "--prefix", type=str, default="image",
                help="output_filename_prefix")
args = vars(ap.parse_args())
# %% ##################################################################
# load the input image,covert to Numpy array and then reshape it
# to have one extra dimension
logger.info(f"loading example image")
image = load_img(path=args["image"])
image = img_to_array(image)  # type: np.ndarray
image = np.expand_dims(image, axis=0)
# %% ##################################################################
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")
total = 0
logger.info(f"generating images")

image_gen = aug.flow(image,
                     batch_size=1,
                     save_to_dir=args["output"],
                     save_prefix=args["prefix"],
                     save_format="jpg")
# loop over examples from our image augmentation generator
for image in image_gen:
    total += 1
    if total == 10:
        break
# %% ##################################################################
