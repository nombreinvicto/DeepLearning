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

sns.set()
# %%
# initialise the class labels
classLabels = ['3T3', 'MG63']

dataFolder = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\cellImages"

args = {
    'dataset': dataFolder
}

imagePaths = list(paths.list_images(args['dataset']))
imagePaths = np.array(imagePaths)
idxs = np.random.randint(0, len(imagePaths), size=(20,))
imagePaths = imagePaths[idxs]
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
save_dir = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\PyCharm Projects\DeepLearningCV\keras_dir\Project_CellTypeDetection\checkpoint"
model = load_model(save_dir)
preds = model.predict(data).argmax(axis=1)
# %%
preds
# %%
for i, imagePath in enumerate(imagePaths):
    image = cv2.imread(imagePath)

    resized = cv2.resize(image.copy(), dsize=(500, 500))
    cv2.putText(resized, f"Label: {classLabels[preds[i]]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", resized)
    cv2.waitKey(0)
# %%
