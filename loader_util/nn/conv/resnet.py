import sys, os, json
import numpy as np
import pandas as pd
import seaborn as sns
import argparse, progressbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes

sns.set()
# %%
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, \
    BatchNormalization, MaxPooling2D, AveragePooling2D, Dropout
from tensorflow.keras.layers import Input, concatenate, add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model


# %%

class Resnet:
    @staticmethod
    def residual_module(data, K, stride, chand_dim, red=False,
                        reg=0.0001, bn_eps=2e-5, bn_mom=0.9):
        shortcut = data

        # first block of 1x1 conv
        bn1 = BatchNormalization(axis=chand_dim, epsilon=bn_eps,
                                 momentum=bn_mom)(data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(filters=int(K / 4), kernel_size=(1, 1),
                       use_bias=False, kernel_regularizer=l2(reg))(act1)

        # second block of 3x3
        bn2 = BatchNormalization(axis=chand_dim, epsilon=bn_eps,
                                 momentum=bn_mom)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(filters=int(K / 4), kernel_size=(3, 3),
                       strides=stride, padding='same',
                       use_bias=False, kernel_regularizer=l2(reg))(act2)

        # third block of 1x1
        bn3 = BatchNormalization(axis=chand_dim, epsilon=bn_eps,
                                 momentum=bn_mom)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(filters=K, kernel_size=(1, 1), use_bias=False,
                       kernel_regularizer=l2(reg)(act3))

        if red:
            shortcut = Conv2D(filters=K, kernel_size=(1, 1), strides=stride,
                              use_bias=False, kernel_regularizer=l2(reg)(act1))

        # add together the shortcut and final CONV
        x = add([conv3, shortcut])
        return x

    @staticmethod
    def build(width, height, depth, classes, stages: list, filters: list,
              reg=0.0001, bn_eps=2e-5, bn_mom=0.9, dataset='cifar'):
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == 'channel_first':
            input_shape = depth, height, width
            chan_dim = 1

        # set input and apply BN
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chan_dim,
                               epsilon=bn_eps,
                               momentum=bn_mom)(inputs)
        if dataset == 'cifar':
            # apply single CONV layer
            x = Conv2D(filters=filters[0], kernel_size=(3, 3), use_bias=False,
                       padding='same', kernel_regularizer=l2(reg))(x)

            for i in range(len(stages)):
                pass
