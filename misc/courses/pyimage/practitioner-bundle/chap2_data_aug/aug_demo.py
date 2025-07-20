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

# load an image in PIL format as RGB
image = load_img(args["image"])

# convert image to numpy array
image = img_to_array(image)

# adding batch axis to image since
# tflow expects that dimension
image = np.expand_dims(image, axis=0)

# lets init the augmentor.
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')
# %% ##################################################################
print(f"[INFO] generating aumented images......")

# this is a generator so will need to iterate over it
image_gen = aug.flow(image,
                     batch_size=1,
                     save_to_dir=args["output"],
                     save_prefix=args["prefix"],
                     save_format='jpg')
total = 0
for image in image_gen:
    total += 1
    if total == 10:
        break
# %% ##################################################################
