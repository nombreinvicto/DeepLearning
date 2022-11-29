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
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# %% ##################################################################
class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, num_filters, fx, fy, stride, chan_dim, padding='same'):
        # define a CONV => BN => RELU pattern
        # notice that BN is before Activation
        x = Conv2D(filters=num_filters,
                   kernel_size=(fx, fy),
                   strides=stride,
                   padding=padding)(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def inception_module(x, num_filters1, num_filters3, chan_dim):
        conv1 = MiniGoogLeNet.conv_module(x=x,
                                          num_filters=num_filters1,
                                          fx=1,
                                          fy=1,
                                          stride=(1, 1),
                                          chan_dim=chan_dim)
        conv3 = MiniGoogLeNet.conv_module(x=x,
                                          num_filters=num_filters3,
                                          fx=3,
                                          fy=3,
                                          stride=(1, 1),
                                          chan_dim=chan_dim)
        x = concatenate([conv1, conv3], axis=chan_dim)
        return x

    @staticmethod
    def downsample_module(x, num_filters, chan_dim):
        conv3 = MiniGoogLeNet.conv_module(x=x,
                                          num_filters=num_filters,
                                          fx=3,
                                          fy=3,
                                          stride=(2, 2),
                                          chan_dim=chan_dim,
                                          padding="valid")
        pool = MaxPooling2D(pool_size=(3, 3),
                            strides=(2, 2))(x)
        x = concatenate([conv3, pool], axis=chan_dim)
        return x

    @staticmethod
    def build(width, height, depth, classes):
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = depth, height, width
            chan_dim = 1

        # define the model input
        inputs = Input(shape=input_shape)
        x = MiniGoogLeNet.conv_module(x=inputs,
                                      num_filters=96,
                                      fx=3,
                                      fy=3,
                                      stride=(1, 1),
                                      chan_dim=chan_dim)

        # stack 2 inception modules followed by downsampling
        x = MiniGoogLeNet.inception_module(x=x,
                                           num_filters1=32,
                                           num_filters3=32,
                                           chan_dim=chan_dim)

        # stack four inception followed by downsample
        x = MiniGoogLeNet.inception_module(x=x,
                                           num_filters1=112,
                                           num_filters3=48,
                                           chan_dim=chan_dim)
        x = MiniGoogLeNet.inception_module(x=x,
                                           num_filters1=96,
                                           num_filters3=64,
                                           chan_dim=chan_dim)
        x = MiniGoogLeNet.inception_module(x=x,
                                           num_filters1=80,
                                           num_filters3=80,
                                           chan_dim=chan_dim)
        x = MiniGoogLeNet.inception_module(x=x,
                                           num_filters1=48,
                                           num_filters3=96,
                                           chan_dim=chan_dim)

        # stack 2 more inception modules followed by global pool
        x = MiniGoogLeNet.inception_module(x=x,
                                           num_filters1=176,
                                           num_filters3=160,
                                           chan_dim=chan_dim)
        x = MiniGoogLeNet.inception_module(x=x,
                                           num_filters1=176,
                                           num_filters3=160,
                                           chan_dim=chan_dim)
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Dropout(0.5)(x)

        # softmax
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="googlenet")
        return model
