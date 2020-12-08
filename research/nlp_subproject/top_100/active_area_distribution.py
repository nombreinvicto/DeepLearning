from imutils import paths
import numpy as np
import progressbar
from cv2 import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes
sns.set()

#%%
imagePath = r"C:\Users\mhasa\Desktop\mvcnn_reorg"
trainPaths = list(paths.list_images(imagePath))
#%%

shape_array = np.zeros(shape=(1, len(trainPaths)))
percent_array = np.zeros(shape=(1, len(trainPaths)))

# imag = cv2.imread(trainPaths[112], cv2.IMREAD_GRAYSCALE)
# cv2.imshow('output', imag)
# print(imag.shape)
# print(imag.max())
# print(imag.min())
#
#
# cv2.waitKey(0)


for i, path in enumerate(trainPaths):

    # read the image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # populate the shape arra
    shape_array[0][i] = img.shape[0]

    # binarise the image
    bin_img = (img != 255) # returns an image array where non w

    # sum the booleans
    bin_img_sum = bin_img.sum() # no of pixels that aint white

    # calculate proportion
    percent = bin_img_sum / (img.shape[0] ** 2)

    # populate the array
    percent_array[0][i] = percent

#%%
sns.distplot(percent_array, hist=True, color="blue")
plt.xlabel("Active Area Proportion")
plt.show()
#%%
sns.distplot(shape_array, hist=True)
plt.show()
