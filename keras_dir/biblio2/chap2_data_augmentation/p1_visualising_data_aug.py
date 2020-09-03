#%%
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loader_util.preprocessing import ImageToArrayPreprocessor, AspectAwarePreprocessor
from loader_util.datasets import SimpleDatasetLoader
from loader_util.nn.conv import FCHeadNet
##
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from loader_util.preprocessing import ImageToArrayPreprocessor
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import argparse
from cv2 import cv2
import numpy as np
from imutils import paths
#%%

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the input image")
# ap.add_argument("-o", "--output", required=True, help="path to output")
# ap.add_argument("-p", "--prefix", type=str, default="image",
#                 help="output filename prefix")
# args = vars(ap.parse_args())

args = {
    "image": r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm "
             r"Projects\DeepLearningCV\keras_dir\biblio2\chap2\mountain.jpg",
    "output": r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm "
              r"Projects\DeepLearningCV\keras_dir\biblio2\chap2\out"
}
#%%

# load the input image
print(f"[INFO] loading example image.....")




# iap = ImageToArrayPreprocessor() # supposed to convert PIL to 3d numpy array
# image = cv2.imread(args['image'])
# image = iap.preprocess(image)



image = load_img(args["image"])
image = img_to_array(image)





image = np.expand_dims(image, axis=0)
print(image.shape)

#%%

# construct the image generator
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

total = 0
#%%

# construct the generator
imgGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
                  save_prefix="image", save_format="jpg")

for image in imgGen:
    total += 1
    if total == 10:
        break

#%%