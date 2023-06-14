# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K


# %% ##################################################################
class MiniGooglenet:
    @staticmethod
    def conv_module(x,
                    filters,
                    kx, ky,
                    strides,
                    chan_dim,
                    padding="same"):
        x = Conv2D(filters=filters,
                   kernel_size=(kx, ky),
                   strides=strides,
                   padding=padding)(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation("relu")(x)

        return x

    @staticmethod
    def inception_module(x,
                         filters1,
                         filters3,
                         chan_dim):
        conv1 = MiniGooglenet.conv_module(x=x,
                                          filters=filters1,
                                          kx=1,
                                          ky=1,
                                          strides=(1, 1),
                                          chan_dim=chan_dim)
        conv3 = MiniGooglenet.conv_module(x=x,
                                          filters=filters3,
                                          kx=3,
                                          ky=3,
                                          strides=(1, 1),
                                          chan_dim=chan_dim)
        x = concatenate([conv1, conv3], axis=chan_dim)

        return x

    @staticmethod
    def down_sample(x,
                    filters,
                    chan_dim):
        conv3 = MiniGooglenet.conv_module(x=x,
                                          filters=filters,
                                          kx=3,
                                          ky=3,
                                          strides=(2, 2),
                                          chan_dim=chan_dim,
                                          padding="valid")
        pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = concatenate([conv3, pool], axis=chan_dim)

        return x

    @staticmethod
    def build(width, height,
              depth, classes):
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = depth, height, width
            chan_dim = 1

        # define the model input
        # 32 x 32 x 3
        inputs = Input(shape=input_shape)

        # 32 x 32 x 3 => 32 x 32 x 96
        x = MiniGooglenet.conv_module(x=inputs,
                                      filters=96,
                                      kx=3,
                                      ky=3,
                                      strides=(1, 1),
                                      chan_dim=chan_dim)

        # %% ##################################################################
        # 2 inception modules followed by downsample
        # 32 x 32 x 96 => 32 x 32 x 64
        x = MiniGooglenet.inception_module(x=x,
                                           filters1=32,
                                           filters3=32,
                                           chan_dim=chan_dim)

        # 32 x 32 x 64 => 32 x 32 x 80
        x = MiniGooglenet.inception_module(x=x,
                                           filters1=32,
                                           filters3=48,
                                           chan_dim=chan_dim)
        # 32 x 32 x 80 => 15 x 15 x 160
        x = MiniGooglenet.down_sample(x=x,
                                      filters=80,
                                      chan_dim=chan_dim)
        # %% ##################################################################
        # 4 inception modules followed by downsample module
        x = MiniGooglenet.inception_module(x=x,
                                           filters1=112,
                                           filters3=48,
                                           chan_dim=chan_dim)
        x = MiniGooglenet.inception_module(x=x,
                                           filters1=96,
                                           filters3=64,
                                           chan_dim=chan_dim)
        x = MiniGooglenet.inception_module(x=x,
                                           filters1=80,
                                           filters3=80,
                                           chan_dim=chan_dim)
        # 15 x 15 x 160 => 15 x 15 x 144
        x = MiniGooglenet.inception_module(x=x,
                                           filters1=48,
                                           filters3=96,
                                           chan_dim=chan_dim)
        # 15 x 15 x 144 => 7 x 7 x 240
        x = MiniGooglenet.down_sample(x=x,
                                      filters=96,
                                      chan_dim=chan_dim)
        # %% ##################################################################
        # 2 more inception modules then global pool and dropout
        # 7 x 7 x 240 => 7 x 7 x 336
        x = MiniGooglenet.inception_module(x=x,
                                           filters1=176,
                                           filters3=160,
                                           chan_dim=chan_dim)
        # 7 x 7 x 336 => 7 x 7 x 336
        x = MiniGooglenet.inception_module(x=x,
                                           filters1=176,
                                           filters3=160,
                                           chan_dim=chan_dim)
        # 7 x 7 x 336 => 1 x 1 x 336
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create a model
        model = Model(inputs, x, name="minigooglenet")
        return model
