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
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# %% ##################################################################
class DeeperGooglenet:
    @staticmethod
    def conv_module(x,
                    filters,
                    kx, ky,
                    strides,
                    chan_dim,
                    padding="same",
                    reg=0.0005,
                    name=None):
        # init the layer names
        conv_name, bn_name, act_name = None, None, None

        # if layer name supplied, then prepend it
        if name:
            conv_name = name + "_conv"
            bn_name = name + "_bn"
            act_name = name + "_act"

        # define the cnv -> bn -> relu pattern
        x = Conv2D(filters=filters,
                   kernel_size=(kx, ky),
                   strides=strides,
                   padding=padding,
                   kernel_regularizer=l2(reg),
                   name=conv_name)(x)
        x = BatchNormalization(axis=chan_dim, name=bn_name)(x)
        x = Activation("relu", name=act_name)(x)

        return x

    @staticmethod
    def inception_module(x,
                         filters1,
                         filters3red,
                         filters3,
                         filters5red,
                         filters5,
                         filters1proj,
                         chan_dim,
                         stage,
                         reg=0.0005):
        first = DeeperGooglenet.conv_module(x=x,
                                            filters=filters1,
                                            kx=1, ky=1,
                                            strides=(1, 1),
                                            chan_dim=chan_dim,
                                            reg=reg,
                                            name=stage + "_first")
        # %% ##################################################################
        second = DeeperGooglenet.conv_module(x=x,
                                             filters=filters3red,
                                             kx=1, ky=1,
                                             strides=(1, 1),
                                             chan_dim=chan_dim,
                                             reg=reg,
                                             name=stage + "_second1")
        second = DeeperGooglenet.conv_module(x=second,
                                             filters=filters3,
                                             kx=3, ky=3,
                                             strides=(1, 1),
                                             chan_dim=chan_dim,
                                             reg=reg,
                                             name=stage + "_second2")
        # %% ##################################################################
        # define the third branch of the Inception module which
        # are our 1x1 and 5x5 convolutions
        third = DeeperGooglenet.conv_module(x,
                                            filters=filters5red,
                                            kx=1, ky=1,
                                            strides=(1, 1),
                                            chan_dim=chan_dim,
                                            reg=reg,
                                            name=stage + "_third1")
        third = DeeperGooglenet.conv_module(third,
                                            filters=filters5,
                                            kx=5, ky=5,
                                            strides=(1, 1),
                                            chan_dim=chan_dim,
                                            reg=reg,
                                            name=stage + "_third2")
        # %% ##################################################################
        # define the fourth branch of the Inception module which
        # is the POOL projection
        fourth = MaxPooling2D((3, 3),
                              strides=(1, 1),
                              padding="same",
                              name=stage + "_pool")(x)
        fourth = DeeperGooglenet.conv_module(fourth,
                                             filters=filters1proj,
                                             kx=1, ky=1,
                                             strides=(1, 1),
                                             chan_dim=chan_dim,
                                             reg=reg,
                                             name=stage + "_fourth")
        # %% ##################################################################
        # concatenate across the channel dimension
        x = concatenate([first, second, third, fourth],
                        axis=chan_dim,
                        name=stage + "_mixed")

        # return the block
        return x

    @staticmethod
    def build(width, height, depth, classes, reg=0.0005):
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == "channel_first"
            input_shape = depth, height, width
            chan_dim = 1

        inputs = Input(shape=input_shape)
        x = DeeperGooglenet.conv_module(inputs,
                                        filters=64,
                                        kx=5,
                                        ky=5,
                                        strides=(1, 1),
                                        chan_dim=chan_dim,
                                        reg=reg,
                                        name="block1")
        x = MaxPooling2D((3, 3),
                         strides=(2, 2),
                         padding="same",
                         name="pool1")(x)

        x = DeeperGooglenet.conv_module(x,
                                        64,
                                        1, 1,
                                        (1, 1),
                                        chan_dim=chan_dim,
                                        reg=reg,
                                        name="block2")

        x = DeeperGooglenet.conv_module(x,
                                        192,
                                        3, 3,
                                        (1, 1),
                                        chan_dim=chan_dim,
                                        reg=reg,
                                        name="block3")

        x = MaxPooling2D((3, 3),
                         strides=(2, 2),
                         padding="same",
                         name="pool2")(x)
