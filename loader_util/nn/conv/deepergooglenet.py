# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf


# %% ##################################################################
class DeeperGoogleNet:
    @staticmethod
    def conv_module(x,
                    num_filters,
                    fx,
                    fy,
                    stride,
                    chan_dim,
                    padding='same',
                    reg=0.0005,
                    name=None):
        # init the layer name suffixes
        convname, bnname, actname = None, None, None

        # if a layer name was supplied prepend it
        if name:
            convname = name + "_conv"
            bnname = name + "_bn"
            actname = name + "_act"

        # define a CONV => BN => RELU pattern
        # here we have BA pattern and NOT AB(P) pattern
        x = Conv2D(filters=num_filters,
                   kernel_size=(fx, fy),
                   strides=stride,
                   padding=padding,
                   kernel_regularizer=l2(reg),
                   name=convname)((x))
        x = BatchNormalization(axis=chan_dim, name=bnname)(x)
        x = Activation("relu", name=actname)(x)

        return x

    @staticmethod
    def inception_module(x,
                         num1_filters,
                         num3_reduce_filters,
                         num3_filters,
                         num5_reduce_filters,
                         num5_filters,
                         num1_proj_filters,
                         chan_dim,
                         stage,
                         reg=0.0005):
        # define the first branch of the Inception module
        # which consists of 1x1 convolutions
        first = DeeperGoogleNet.conv_module(x=x,
                                            num_filters=num1_filters,
                                            fx=1,
                                            fy=1,
                                            stride=(1, 1),
                                            chan_dim=chan_dim,
                                            reg=reg,
                                            name=stage + "_first")

        # define the second branch
        second = DeeperGoogleNet.conv_module(x=x,
                                             num_filters=num3_reduce_filters,
                                             fx=1,
                                             fy=1,
                                             stride=(1, 1),
                                             chan_dim=chan_dim,
                                             reg=reg,
                                             name=stage + "_second1")
        second = DeeperGoogleNet.conv_module(x=second,
                                             num_filters=num3_filters,
                                             fx=3,
                                             fy=3,
                                             stride=(1, 1),
                                             chan_dim=chan_dim,
                                             reg=reg,
                                             name=stage + "_second2")

        # define the third branch
        third = DeeperGoogleNet.conv_module(x=x,
                                            num_filters=num5_reduce_filters,
                                            fx=1,
                                            fy=1,
                                            stride=(1, 1),
                                            chan_dim=chan_dim,
                                            reg=reg,
                                            name=stage + "_third1")
        third = DeeperGoogleNet.conv_module(x=third,
                                            num_filters=num5_filters,
                                            fx=5,
                                            fy=5,
                                            stride=(1, 1),
                                            chan_dim=chan_dim,
                                            reg=reg,
                                            name=stage + "_third2")

        # define the fourth branch
        fourth = MaxPooling2D(pool_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              name=stage + "_pool")(x)
        fourth = DeeperGoogleNet.conv_module(x=fourth,
                                             num_filters=num1_proj_filters,
                                             fx=1,
                                             fy=1,
                                             stride=(1, 1),
                                             chan_dim=chan_dim,
                                             reg=reg,
                                             name=stage + "_fourth")

        # concatenate the outputs of the inception module
        x = concatenate([first, second, third, fourth],
                        axis=chan_dim,
                        name=stage + "_mixed")
        return x

    @staticmethod
    def build(width,
              height,
              depth,
              classes,
              reg=0.0005):
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == "channel_first":
            input_shape = depth, height, width
            chan_dim = 1

        # lets build the network
        # say, input = 64x64x3
        inputs = Input(shape=input_shape)

        # out = 64x64x64
        x = DeeperGoogleNet.conv_module(x=inputs,
                                        num_filters=64,
                                        fx=5,
                                        fy=5,
                                        stride=(1, 1),
                                        chan_dim=chan_dim,
                                        reg=reg,
                                        name="block1")

        # out = 32x32x64
        x = MaxPooling2D(pool_size=(3, 3),
                         strides=(2, 2),
                         padding="same",
                         name="pool1")(x)

        # out = 32x32x64
        x = DeeperGoogleNet.conv_module(x=x,
                                        num_filters=64,
                                        fx=1,
                                        fy=1,
                                        stride=(1, 1),
                                        chan_dim=chan_dim,
                                        reg=reg,
                                        name="block2")

        # out = 32x32x192
        x = DeeperGoogleNet.conv_module(x=x,
                                        num_filters=192,
                                        fx=3,
                                        fy=3,
                                        stride=(1, 1),
                                        chan_dim=chan_dim,
                                        reg=reg,
                                        name="block3")

        # out 16x16x192
        x = MaxPooling2D(pool_size=(3, 3),
                         strides=(2, 2),
                         padding="same",
                         name="pool2wp")(x)

        # apply 2 inception modules followed by pool
        # out = 16x16x(64+128+32+32) = 16x16x256
        x = DeeperGoogleNet.inception_module(x=x,
                                             num1_filters=64,
                                             num3_reduce_filters=96,
                                             num3_filters=128,
                                             num5_reduce_filters=16,
                                             num5_filters=32,
                                             num1_proj_filters=32,
                                             chan_dim=chan_dim,
                                             stage="3a",
                                             reg=reg)

        # out = 16x16x(128+192+96+64) = 16x16x480
        x = DeeperGoogleNet.inception_module(x=x,
                                             num1_filters=128,
                                             num3_reduce_filters=128,
                                             num3_filters=192,
                                             num5_reduce_filters=32,
                                             num5_filters=96,
                                             num1_proj_filters=64,
                                             chan_dim=chan_dim,
                                             stage="3b",
                                             reg=reg)
        # out = 8x8x480
        x = MaxPooling2D(pool_size=(3, 3),
                         strides=(2, 2),
                         padding="same",
                         name="pool3")(x)

        # apply 5 inceptions followed by a pool
        # out = 4x4x832
        x = DeeperGoogleNet.inception_module(x, 192, 96, 208, 16,
                                             48, 64, chan_dim, "4a", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 160, 112, 224, 24,
                                             64, 64, chan_dim, "4b", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 128, 128, 256, 24,
                                             64, 64, chan_dim, "4c", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 112, 144, 288, 32,
                                             64, 64, chan_dim, "4d", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 256, 160, 320, 32,
                                             128, 128, chan_dim, "4e", reg=reg)

        x = MaxPooling2D((3, 3),
                         strides=(2, 2),
                         padding="same",
                         name="pool4")(x)

        # finally apply average pooling and dropout
        x = AveragePooling2D(pool_size=(4, 4),
                             name="pool5")(x)
        x = Dropout(0.4, name="do")(x)

        # softmax classifier
        x = Flatten(name="flatten")(x)
        x = Dense(classes, kernel_regularizer=l2(reg), name="labels")(x)
        x = Activation("softmax", name="softmax")(x)

        # return the model
        model = Model(inputs, x, name="googlenet")
        return model
