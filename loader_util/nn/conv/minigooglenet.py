# import the necessary packages
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    AveragePooling2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, \
    Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class MiniGoogleNet:
    @staticmethod
    def conv_module(x, K, kx, ky, stride, chanDim, padding="same"):
        # define a CONV => BN => RELU pattern
        x = Conv2D(filters=K, kernel_size=(kx, ky), strides=stride,
                   padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation('relu')(x)

        # return the block
        return x

    @staticmethod
    def inception_module(x, numk1x1, numk3x3, chanDim):
        # define two CONV modules then concatenate across the channel dims
        conv1x1 = MiniGoogleNet.conv_module(x=x,
                                            K=numk1x1,
                                            kx=1,
                                            ky=1,
                                            stride=(1, 1),
                                            chanDim=chanDim)
        conv3x3 = MiniGoogleNet.conv_module(x=x,
                                            K=numk3x3,
                                            kx=3,
                                            ky=3,
                                            stride=(1, 1),
                                            chanDim=chanDim)
        x = concatenate([conv1x1, conv3x3], axis=chanDim)

        # return the block
        return x

    @staticmethod
    def downsample_module(x, K, chanDim):
        # define the CONV module and POOL, then concatenate across the
        # channel dimensions
        conv_3x3 = MiniGoogleNet.conv_module(x=x,
                                             K=K,
                                             kx=3,
                                             ky=3,
                                             stride=(2, 2),
                                             chanDim=chanDim,
                                             padding="valid")
        pool = MaxPooling2D(pool_size=(3, 3),
                            strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chanDim)

        # return the block
        return x

    @staticmethod
    def build(width, height, depth, classes):
        # initialise the input shape to be channels last
        input_shape = height, width, depth
        chanDim = -1

        if K.image_data_format() == 'channels_first':
            input_shape = depth, height, width
            chanDim = 1

        # define the model input and first CONV module
        inputs = Input(shape=input_shape)
        x = MiniGoogleNet.conv_module(inputs,
                                      K=96,  # no of filter
                                      kx=3,  # filter size x
                                      ky=3,  # filter size y
                                      stride=(1, 1),
                                      chanDim=chanDim)

        # two inception modules followed by downsample
        x = MiniGoogleNet.inception_module(x=x,
                                           numk1x1=32,
                                           numk3x3=32,
                                           chanDim=chanDim)

        x = MiniGoogleNet.inception_module(x=x,
                                           numk1x1=32,
                                           numk3x3=48,
                                           chanDim=chanDim)
        x = MiniGoogleNet.downsample_module(x=x,
                                            K=80,
                                            chanDim=chanDim)

        # four Inception modules followed by a downsample module
        x = MiniGoogleNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogleNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogleNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogleNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogleNet.downsample_module(x, 96, chanDim)

        # two Inception modules followed by global POOL and dropout
        x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="googlenet")

        # return the constructed network architecture
        return model
