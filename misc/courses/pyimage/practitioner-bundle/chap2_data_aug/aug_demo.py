from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse

# %% ##################################################################
# contruct the argument parser and parse
ap = argparse.ArgumentParser()
ap.add_argument("-i",
                "--image",
                required=True,
                help="path to the input image")
ap.add_argument("-o",
                "--output",
                required=True,
                help="path to output dir to store the augmented image")
ap.add_argument("-p",
                "--prefix",
                type=str,
                default="image",
                help="output filename prefix")
args = vars(ap.parse_args())
# %% ##################################################################
print(f"[INFO] loading image here......")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
