# import the necessary packages
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    AveragePooling2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, \
    Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# %%

class DeeperGoogleNet:
    @staticmethod
    def conv_module(x, K, kx, ky, stride, chanDim, padding='same',
                    reg=0.0005, name=None):
        # initialise the CONV, BN and RELU layer names
        convName, bnName, actName = None, None, None

        # if a layer name was supplied prepend it
        if name is not None:
            convName = name + "_conv"
            bnName = name + "_bn"
            actName = name + "_act"

        # define a CONV => BN => RELU pattern
        x = Conv2D(filters=K, kernel_size=(kx, ky), strides=stride,
                   padding=padding, kernel_regularizer=l2(reg),
                   name=convName)(x)
        x = BatchNormalization(axis=chanDim, name=bnName)(x)
        x = Activation(activation="relu", name=actName)(x)

        # return the tensor
        return x

    @staticmethod
    def inception_module(x,
                         num1x1,
                         num3x3Reduce,
                         num3x3,
                         num5x5Reduce,
                         num5x5,
                         num1x1Proj,
                         chanDim,
                         stage,
                         reg=0.0005):
        # define the first branch of the Inception module which consists of
        # 1 x 1 convolutions
        first = DeeperGoogleNet.conv_module(x, num1x1, kx=1, ky=1,
                                            stride=(1, 1), chanDim=chanDim,
                                            name=stage + "_first")

        # define the second branch of the Inception module which consists of
        # 1x1 and 3x3 convolutions
        second = DeeperGoogleNet.conv_module(x, num3x3Reduce, 1, 1,
                                             (1, 1), chanDim, reg=reg,
                                             name=stage + "_second1")
        second = DeeperGoogleNet.conv_module(second, num3x3, 3, 3,
                                             (1, 1), chanDim, reg=reg,
                                             name=stage + "_second2")

        # define the third branch of the Inception module which are our 1x1
        # and 5x5 convolutions
        third = DeeperGoogleNet.conv_module(x, num5x5Reduce, 1, 1, (1, 1),
                                            chanDim, reg=reg,
                                            name=stage + "_third1")
        third = DeeperGoogleNet.conv_module(third, num5x5, 5, 5, (1, 1),
                                            chanDim,
                                            reg=reg,
                                            name=stage + "_third2")

        # define the fourth branch of the Inception module which is the POOL
        # projection
        fourth = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same",
                              name=stage + "_pool")(x)
        fourth = DeeperGoogleNet.conv_module(fourth, num1x1Proj, 1, 1, (1, 1),
                                             chanDim, reg=reg,
                                             name=stage + "_fourth")

        # concatenate across the channel dims
        x = concatenate([first, second, third, fourth], axis=chanDim,
                        name=stage + "_mixed")

        # return the tensor
        return x

    @staticmethod
    def build(width,
              height,
              depth,
              classes,
              reg=0.0005):
        # init the input shape to be channel last and the channels dim itself
        inputShape = height, width, depth
        chanDim = -1

        # if we are using channel first update the input shape
        if K.image_data_format() == "channel_first":
            inputShape = depth, height, width

        # define the model input, followed by sequence of CONV => POOL => (
        # CONV*2) => POOL layers
        inputs = Input(shape=inputShape)
        x = DeeperGoogleNet.conv_module(inputs, K=64, kx=5, ky=5,
                                        stride=(1, 1), chanDim=chanDim,
                                        reg=reg, name="block1")
        x = MaxPooling2D((3, 3), strides=(2, 2),
                         padding="same", name="pool1")(x)
        x = DeeperGoogleNet.conv_module(x, 64, 1, 1, (1, 1), chanDim,
                                        reg=reg, name="block2")
        x = DeeperGoogleNet.conv_module(x, 192, 3, 3, (1, 1), chanDim,
                                        reg=reg, name="block3")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
                         name="pool2")(x)

        # apply 2 Inception
        # apply two Inception modules followed by a POOL
        x = DeeperGoogleNet.inception_module(x, 64, 96, 128, 16, 32, 32,
                                             chanDim, "3a",
                                             reg=reg)
        x = DeeperGoogleNet.inception_module(x, 128, 128, 192, 32, 96, 64,
                                             chanDim, "3b", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
                         name="pool3")(x)

        # apply five Inception modules followed by POOL
        x = DeeperGoogleNet.inception_module(x, 192, 96, 208, 16, 48, 64,
                                             chanDim, "4a", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 160, 112, 224, 24, 64, 64,
                                             chanDim, "4b", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 128, 128, 256, 24, 64, 64,
                                             chanDim, "4c", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 112, 144, 288, 32, 64, 64,
                                             chanDim, "4d", reg=reg)
        x = DeeperGoogleNet.inception_module(x, 256, 160, 320, 32, 128, 128,
                                             chanDim, "4e", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same",
                         name="pool4")(x)

        # apply a POOL layer (average) followed by dropout
        x = AveragePooling2D((4, 4), name="pool5")(x)
        x = Dropout(0.4, name="do")(x)

        # softmax classifier x = Flatten(name="flatten")(x)
        x = Dense(classes, kernel_regularizer=l2(reg), name="labels")(x)
        x = Activation("softmax", name="softmax")(x)

        # create the model
        model = Model(inputs, x, name="googlenet")

        # return the constructed network architecture
        return model
