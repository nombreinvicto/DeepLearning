# import necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from loader_util.utils.captchahelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import imutils
import cv2

sns.set()
# %%

args = {
    'input': '',
    'model': ''
}

# load pretrained network
print('[INFO] loading pretrained network......')
model = load_model(args['model'])

# randomly sample a few of the input images
imagePaths = list(paths.list_images(args['input']))
imagePaths = np.random.choice(imagePaths, size=(10, 0), replace=False)

# %%
# loop over the image paths
for imagePath in imagePaths:
    # load the image and convert it to grayscale
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold the image to reveal the digits
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

