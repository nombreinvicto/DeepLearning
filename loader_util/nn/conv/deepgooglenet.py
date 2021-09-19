import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, \
    BatchNormalization, MaxPooling2D, AveragePooling2D, Dropout
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# %%

class DeepGoogleNet:
    @staticmethod
    def conv_module(x, k, kx, ky, stride,
                    chan_dim, padding='same',
                    reg=0.005,
                    name=None):
        # init the CONV, BN, RELU layer names
        conv_name, bn_name, act_name = None, None, None

        # if a layer name waws supplied, prepend it
        if name:
            conv_name = name + "_conv"
            bn_name = name + "_bn"
            act_name = name + "_act"

        # define a conv then bn then act patters
        x = Conv2D(filters=k, kernel_size=(kx, ky),
                   strides=stride, padding=padding,
                   kernel_regularizer=l2(reg), name=conv_name)(x)
        x = BatchNormalization(axis=chan_dim, name=bn_name)(x)
        x = Activation("relu", name=act_name)(x)

        return x

    @staticmethod
    def inception_module(x, numx1, numx3_reduce, numx3,
                         numx5_reduce, numx5, numx1_proj, chan_dim, stage,
                         reg=0.0005):
        # define first branch of inceptions
        first = DeepGoogleNet.conv_module(x, k=numx1, kx=1, ky=1,
                                          stride=(1, 1), chan_dim=chan_dim,
                                          reg=reg, name=stage + "_first")

        # second branch
        second = DeepGoogleNet.conv_module(x, k=numx3_reduce, kx=1, ky=1,
                                           stride=(1, 1), chan_dim=chan_dim,
                                           reg=reg, name=stage + "_second1")

        second = DeepGoogleNet.conv_module(second, k=numx3, kx=3, ky=3,
                                           stride=(1, 1), chan_dim=chan_dim,
                                           reg=reg, name=stage + "_second2")

        # third branch
        third = DeepGoogleNet.conv_module(x, k=numx5_reduce, kx=1, ky=1,
                                          stride=(1, 1), chan_dim=chan_dim,
                                          reg=reg, name=stage + "_third1")

        third = DeepGoogleNet.conv_module(third, k=numx5, kx=5, ky=5,
                                          stride=(1, 1), chan_dim=chan_dim,
                                          reg=reg, name=stage + "_third2")

        # fourth branch
        fourth = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same',
                              name=stage + "_fourth_pool")(x)

        fourth = DeepGoogleNet.conv_module(fourth, k=numx1_proj, kx=1, ky=1,
                                           stride=(1, 1), chan_dim=chan_dim,
                                           reg=reg, name=stage + "_fourth")

        x = concatenate([first, second, third, fourth], axis=chan_dim,
                        name=stage + "_mixed")
        return x

    @staticmethod
    def build(width, height, depth, classes, reg=0.005):
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == "channel_first":
            input_shape = depth, height, width
            chan_dim = 1

        inputs = Input(shape=input_shape)
        x = DeepGoogleNet.conv_module(inputs, k=64, kx=5, ky=5,
                                      stride=(1, 1), chan_dim=chan_dim,
                                      reg=reg, name="block1")
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         padding='same', name="pool1")(x)
        x = DeepGoogleNet.conv_module(x, k=64, kx=1, ky=1,
                                      stride=(1, 1), chan_dim=chan_dim,
                                      reg=reg, name="block2")
        x = DeepGoogleNet.conv_module(x, k=192, kx=3, ky=3,
                                      stride=(1, 1), chan_dim=chan_dim,
                                      reg=reg, name="block3")
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         padding='same', name="pool2")(x)

        # apply 2 inceptions followed by pool
        x = DeepGoogleNet.inception_module(x, numx1=64,
                                           numx3_reduce=96, numx3=128,
                                           numx5_reduce=16, numx5=32,
                                           numx1_proj=32,
                                           chan_dim=chan_dim, stage="3a",
                                           reg=reg)

        x = DeepGoogleNet.inception_module(x, numx1=128,
                                           numx3_reduce=128, numx3=192,
                                           numx5_reduce=32, numx5=96,
                                           numx1_proj=64,
                                           chan_dim=chan_dim, stage="3b",
                                           reg=reg)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                         name="pool3")(x)

        # apply five Inception modules followed by POOL
        x = DeepGoogleNet.inception_module(x, 192, 96, 208, 16, 48, 64,
                                           chan_dim, "4a", reg=reg)

        x = DeepGoogleNet.inception_module(x, 160, 112, 224, 24,
                                           64, 64, chan_dim, "4b", reg=reg)

        x = DeepGoogleNet.inception_module(x, 128, 128, 256, 24,
                                           64, 64, chan_dim, "4c", reg=reg)

        x = DeepGoogleNet.inception_module(x, 112, 144, 288, 32,
                                           64, 64, chan_dim, "4d", reg=reg)

        x = DeepGoogleNet.inception_module(x, 256, 160, 320, 32,
                                           128, 128, chan_dim, "4e", reg=reg)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
                         name="pool4")(x)

        # apply a POOL layer (average) followed by dropout
        x = AveragePooling2D((4, 4), name="pool5")(x)
        x = Dropout(0.4, name="do")(x)

        # softmax classifier
        x = Flatten(name="flatten")(x)
        x = Dense(classes, kernel_regularizer=l2(reg),
                  name="labels")(x)
        x = Activation("softmax", name="softmax")(x)

        # create the model
        model = Model(inputs, x, name="googlenet")

        # return the constructed network architecture
        return model
# %%
