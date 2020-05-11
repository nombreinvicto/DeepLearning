import sys
import numpy as np
import pandas as pd
import argparse
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
    AspectAwarePreprocessor, MeanSubtractionPreProcessor
from loader_util.datasets import SimpleDatasetLoader
from loader_util.nn.conv import FCHeadNet, ResNet
from loader_util.callbacks import EpochCheckpoint, TrainingMonitor
##
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import cifar10
from imutils import paths

# %%

# construct the argument parser abd parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to specific model chekpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

# %%

# load the train and test data, converting the images from integers to floats
print(f"[INFO] loading CIFAR-10 data.....")
(trainx, trainy), (testx, testy) = cifar10.load_data()
trainx = trainx.astype('float')
testx = testx.astype('float')

# apply mean subtraction to
mp = MeanSubtractionPreProcessor()
trainx_preprocessed = np.zeros_like(trainx)
testx_preprocessed = np.zeros_like(testx)

for i, image in enumerate(trainx):
    trainx_preprocessed[i] = mp.preprocess(image)

for i, image in enumerate(testx):
    testx_preprocessed[i] = mp.preprocess(image)

# %%

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainy = lb.fit_transform(trainy)
testy = lb.transform(testy)

# construct the image generator
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')
# %%

# if there is no specific model checkpoint supplied then initialise the model
if args['model'] is None:
    print(f"[INFO] compiling model ......")
    opt = SGD(lr=1e-1)

    model = ResNet.build(width=32,
                         height=32,
                         depth=3,
                         classes=10,
                         stages=(9, 9, 9),
                         filters=(64,64,128,256), reg=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=['accuracy'])
else:
    # otherwise load the checkpoint frpm disk
    print(f"[INFO] loading: {args['model']}")
    model = load_model(args['model'])

    # update learning rate
    print(f"[INFO] old learning rate: {K.get_value(model.optimizer.lr)}")
    K.set_value(model.optimizer.lr, 1e-5)
    print(f"[INFO] new learning rate: {K.get_value(model.optimizer.lr)}")

#%%

# construct callbacks

callbacks = [
    EpochCheckpoint(args['checkpoints'], every=5,
                    startAt=args["start_epoch"]),
    TrainingMonitor("output/resnet56_cifar10.png",
                    jsonPath="output/resnet56_cifar10.json",
                    startAt=args['start_epoch'])
]

#%%

# train the network

print(f"[INFO] training the network.....")
model.fit_generator(
    aug.flow(trainx_preprocessed, trainy, batch_size=128),
    validation_data=(testx_preprocessed, testy),
    steps_per_epoch=len(trainx_preprocessed) // 128,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

