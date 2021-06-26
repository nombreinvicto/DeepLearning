# %%
import numpy as np
import pandas as pd



# %%
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import argparse as ag

# %%

# construct the argument parrser
ap = ag.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-o', '--output', required=True,
                help='path to output dir to store sumented image')
ap.add_argument('-p', '--prefix', type=str, default='image', help='output '
                                                                  'prefix')
args = vars(ap.parse_args())
# %%
# load image and then reshape
print(f"[INFO] loading example image......")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# %%
# construct the image generator for data augmentation

aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')
total = 0
# %%
# construct the generator
print(f"[INFO] generating images......")
image_gen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
                     save_prefix=args["prefix"], save_format="jpg")

for img in image_gen:
    total += 1

    if total == 10:
        break
#%%


