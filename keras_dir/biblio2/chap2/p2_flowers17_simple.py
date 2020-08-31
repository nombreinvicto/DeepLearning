# import the required packages
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loader_util.preprocessing import ImageToArrayPreprocessor, \
    AspectAwarePreprocessor
from loader_util.datasets import SimpleDatasetLoader
from loader_util.nn.conv import FCHeadNet
##
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from loader_util.nn.conv import MinVGGNet
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from imutils import paths
import numpy as np
import os

# %%

# construct the imagepaths
data_dir = r"C:\Users\mhasa\Google Drive\Tutorial " \
           r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\flowers17\images"

# grab the list of images then extract class labels
print(f"[INFO] loading images.....")
image_paths = list(paths.list_images(basePath=data_dir))
class_names = [pt.split(sep=os.path.sep)[-2] for pt in image_paths]
class_names = [str(x) for x in np.unique(class_names)]

# %%


# initlaise the image preprocessors
aap = AspectAwarePreprocessor(width=64, height=64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk and then scale the pixel intensities [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0
# %%

# partition the data into training and test sets
trainx, testx, trainy, testy = train_test_split(data, labels,
                                                test_size=0.25,
                                                random_state=42)
# convert the labels from the integers
le = LabelBinarizer()
trainy = le.fit_transform(trainy)
testy = le.transform(testy)
# %%

# initialize the optimizer and the model
print(f"[INFO] compiling model.....")
opt = SGD(lr=0.05)
model = MinVGGNet.build(width=64, height=64, depth=3, classes=len(class_names))
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])
# %%

# train the network
print(f"[INFO] training the network .....")
H = model.fit(trainx, trainy,
              validation_data=(testx, testy),
              batch_size=32,
              epochs=100,
              verbose=1)
# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=class_names))

# %%

# plot the performance
epochs = range(1, 101)
loss = H.history['loss']
accuracy = H.history['accuracy']
val_loss = H.history['val_loss']
val_accuracy = H.history['val_accuracy']
plot_df = pd.DataFrame(
    data=np.c_[epochs, loss, accuracy, val_loss, val_accuracy],
    columns=['epochs', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])

# do the actual plots
sns.set(font_scale=1)
f, ax = plt.subplots(1, 1, figsize=(15, 8))
sns.lineplot(data=plot_df, x='epochs', y='loss', ax=ax, label='train loss',
             linewidth=3)
sns.lineplot(data=plot_df, x='epochs', y='accuracy', ax=ax,
             label='train accuracy', linewidth=3)
sns.lineplot(data=plot_df, x='epochs', y='val_loss', ax=ax, label='val loss',
             linewidth=3)
sns.lineplot(data=plot_df, x='epochs', y='val_accuracy', ax=ax,
             label='val_accuracy', linewidth=3)
ax.set_ylabel('Loss or Accuracy')
ax.set_xlabel('Epochs')
plt.setp(ax.get_legend().get_texts(), fontsize='18');  # for legend text
plt.show()
#%%
