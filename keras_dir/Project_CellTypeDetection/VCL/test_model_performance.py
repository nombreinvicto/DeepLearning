# import the necessary packages
from loader_util.preprocessing import ImageToArrayPreprocessor, \
    AspectAwarePreprocessor
from loader_util.datasets import SimpleDatasetLoader
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import cv2
import os

sns.set()
# %%
# initialise the class labels
classLabels = ['3T3', 'MG63', 'hASC']

#dataFolder = r"/home/mhasan3/Desktop/WorkFolder/cellImages2/"
dataFolder = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\DeepLearningCV\keras_dir\Project_CellTypeDetection\VCL\testCellImages"

args = {
    'dataset': dataFolder
}

imagePaths = list(paths.list_images(args['dataset']))
# imagePaths = np.array(imagePaths)
# idxs = np.random.randint(0, len(imagePaths), size=(40,))
# imagePaths = imagePaths[idxs]
# %%
# intialise the preprocessor
sp = AspectAwarePreprocessor(64, 64)
ip = ImageToArrayPreprocessor()

# load dataset and then scale
sdl = SimpleDatasetLoader(preprocessors=[sp, ip])
data, labels = sdl.load(imagePaths)
data = data.astype('float') / 255.0
# %%
# load pretrained model

model = load_model('best_model_weights2.hdf5')
preds = model.predict(data).argmax(axis=1)
# %%
preds
# %%
for i, imagePath in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    actual_label = imagePath.split(os.path.sep)[-2]
    		
    resized = cv2.resize(image.copy(), dsize=(500, 500))
    cv2.putText(resized, f"Pred Label: {classLabels[preds[i]]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(resized, f"Actual Label: {actual_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", resized)
    cv2.waitKey(0)
# %%
