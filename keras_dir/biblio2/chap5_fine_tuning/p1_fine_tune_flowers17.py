# import the necessary packages

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes

sns.set()

# %%
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loader_util.preprocessing import ImageToArrayPreprocessor, \
    AspectAwarePreprocessor
from loader_util.datasets import SimpleDatasetLoader
from loader_util.nn.conv import FCHeadNet
##
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from imutils import paths
import os

# %%

# construct the argument parser
data_dir = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\flowers17\images"

args_dict = {
    "dataset": f"{data_dir}",
    "model": f"{data_dir}//model.hdf5"
}

# %%
# construct the image generator
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')
# %%

# grab the images
print(f"[INFO] loading images.....")
image_paths = list(paths.list_images(args_dict["dataset"]))
class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
class_names = [str(x) for x in np.unique(class_names)]
# %%
print(f"Staring Image preprocessing sequence.....")
# initialise the preprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset then scale
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
data, labels = sdl.load(image_paths, verbose=500)
data = data.astype('float32') / 255.0
print(f"Shape of data: {data.shape}")
print(f"Shape of labels: {labels.shape}")
print("=" * 50)
# %%

trainx, testx, trainy, testy = train_test_split(data, labels, test_size=0.25,
                                                random_state=42)
# transform the labels
le = LabelBinarizer()
trainy = le.fit_transform(trainy)
testy = le.transform(testy)
# %%

# construct the model

base_model = VGG16(weights="imagenet", include_top=False,
                   input_tensor=Input(shape=(224, 224, 3)))  # type: Model

# init the new head model
head_model = FCHeadNet.builld(baseModel=base_model,
                              classes=len(class_names),
                              fc_nodes=[256])

# place fc model on top of base model
model = Model(inputs=base_model.input,
              outputs=head_model)
# %%

# freeze weights in basemodel
for layer in base_model.layers:
    layer.trainable = False

# compile the model
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# %%
epoch_num = 25
batch_size = 32

print(f"Starting training with epoch: {epoch_num} and basemodel weights "
      f"frozen.....")

# now train the compiled model for a few epochs
H = model.fit_generator(aug.flow(trainx, trainy, batch_size=batch_size),
                        validation_data=(testx, testy),
                        epochs=epoch_num,
                        steps_per_epoch=len(trainx) // batch_size,
                        verbose=1)
# %%

# evaluate the network
print(f"[INFO] evaluating model after training....")
predictions = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=class_names))
print("=" * 50)
# %%

# unfreeze final set of conv layers
print(f"Unfreezzing final set of conv layers in base model and "
      f"retraining.....")
for layer in model.layers[15:]:
    layer.trainable = True

# recompile the model
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print(f"Retrain the model with conv layers unfrozen.....")
H = model.fit_generator(aug.flow(trainx, trainy, batch_size=batch_size),
                        validation_data=(testx, testy),
                        epochs=epoch_num + 75,
                        steps_per_epoch=len(trainx) // batch_size,
                        verbose=1)
# %%

print(f"[INFO] evaluating model after fine-tuning....")
predictions = model.predict(testx, batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=class_names))
print("=" * 50)
#%%
print(f"End of Training and Fine tuning.....")

