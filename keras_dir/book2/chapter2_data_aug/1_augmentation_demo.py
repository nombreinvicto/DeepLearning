import logging
import argparse
import numpy as np
# %% ##################################################################
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# %% ##################################################################
# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=True, help="output dir for aumented images")
ap.add_argument("-p", "--prefix", type=str, default="aug", help="output filename prefix")
args = vars(ap.parse_args())
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
logger.info("loading example image")

# load imagae as RGB PIL
image = load_img(args["image"])

# convert RGB PIL image to numpy
image = img_to_array(image)

# add batch dimension to image
image = np.expand_dims(image, axis=0)
# %% ##################################################################
# construct the data augmentor - it is a generator
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

total = 0
# %% ##################################################################
logger.info("generating augmented images")
image_gen = aug.flow(image,
                     batch_size=1,
                     save_to_dir=args["output"],
                     save_prefix=args["prefix"],
                     save_format="jpg")
for aumented_image in image_gen:
    total += 1

    if total == 10:
        break
logger.info("completed data augmentation routine")
# %% ##################################################################
