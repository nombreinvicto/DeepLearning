# %%

# import the necessary packages
from loader_util.nn.conv import FCHeadNet
##
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, \
    BatchNormalization, AveragePooling2D, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# %%l

class MiniGoogleNet:
    @staticmethod
    def conv_module(x, k, kx, ky, stride, chan_dim, padding="same"):
        # define CONV => BA pattern
        x = Conv2D(filters=k,
                   kernel_size=(kx, ky),
                   strides=stride,
                   padding=padding)(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation('relu')(x)

        return x

    # %%
    @staticmethod
    def inception_module(x, numx1, numx3, chan_dim):
        # define 2 CONV modules then concatenate across channel dim
        conv1 = MiniGoogleNet.conv_module(x=x,
                                          k=numx1,
                                          kx=1,
                                          ky=1,
                                          stride=(1, 1),
                                          chan_dim=chan_dim)
        conv3 = MiniGoogleNet.conv_module(x=x,
                                          k=numx3,
                                          kx=3,
                                          ky=3,
                                          stride=(1, 1),
                                          chan_dim=chan_dim)

        x = concatenate([conv1, conv3], axis=chan_dim)
        return x

    # %%
    @staticmethod
    def downsample_module(x, k, chan_dim):
        # define the CONV module and POOL then concatenate across chan dims
        conv3 = MiniGoogleNet.conv_module(x=x,
                                          k=k,
                                          kx=3,
                                          ky=3,
                                          stride=(2, 2),
                                          chan_dim=chan_dim,
                                          padding='valid')
        pool = MaxPooling2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='valid')(x)
        x = concatenate([conv3, pool], axis=chan_dim)
        return x

    # %%
    @staticmethod
    def build(width, height, depth, classes):
        # init the input shape to be the channel last and the channels
        # dimension itself
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == 'channel_first':
            input_shape = depth, height, width
            chan_dim = 1

        # define model input
        inputs = Input(shape=input_shape)
        x = MiniGoogleNet.conv_module(x=inputs,
                                      k=96,
                                      kx=3,
                                      ky=3,
                                      stride=(1, 1),
                                      chan_dim=chan_dim)

        # two inception modules followed by downsample module
        x = MiniGoogleNet.inception_module(x=x,
                                           numx1=32,
                                           numx3=32,
                                           chan_dim=chan_dim)
        x = MiniGoogleNet.inception_module(x=x,
                                           numx1=32,
                                           numx3=48,
                                           chan_dim=chan_dim)
        x = MiniGoogleNet.downsample_module(x=x,
                                            k=80,
                                            chan_dim=chan_dim)

        # 4 inception then downsample
        x = MiniGoogleNet.inception_module(x=x,
                                           numx1=112,
                                           numx3=48,
                                           chan_dim=chan_dim)
        x = MiniGoogleNet.inception_module(x=x,
                                           numx1=96,
                                           numx3=64,
                                           chan_dim=chan_dim)
        x = MiniGoogleNet.inception_module(x=x,
                                           numx1=80,
                                           numx3=80,
                                           chan_dim=chan_dim)
        x = MiniGoogleNet.inception_module(x=x,
                                           numx1=48,
                                           numx3=96,
                                           chan_dim=chan_dim)
        x = MiniGoogleNet.downsample_module(x=x,
                                            k=96,
                                            chan_dim=chan_dim)

        # two inception modules followed by global pool and dropout
        x = MiniGoogleNet.inception_module(x=x,
                                           numx1=176,
                                           numx3=160,
                                           chan_dim=chan_dim)
        x = MiniGoogleNet.inception_module(x=x,
                                           numx1=176,
                                           numx3=160,
                                           chan_dim=chan_dim)
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="googlenet")
        return model


# %%
if __name__ == '__main__':
    sample_model = MiniGoogleNet.build(32, 32, 3, 10)  # type: Model
    output_shape = sample_model.output.shape
    print(output_shape)
