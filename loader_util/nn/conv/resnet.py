# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# %% ##################################################################
class Resnet:
    @staticmethod
    def residual_module(input_activation,
                        filters,
                        strides,
                        chan_dim,
                        red=False,
                        reg=0.0001,
                        bn_eps=2e-5,
                        bn_mom=0.9):
        # shortcut branch
        shortcut = input_activation

        # %% ##################################################################
        # first block 1 x 1 conv
        bn1 = BatchNormalization(axis=chan_dim,
                                 epsilon=bn_eps,
                                 momentum=bn_mom)(input_activation)
        act1 = Activation("relu")(bn1)

        # uses valid padding
        conv1 = Conv2D(int(filters * 0.25),
                       kernel_size=(1, 1),
                       use_bias=False,
                       kernel_regularizer=l2(reg))(act1)
        # %% ##################################################################
        # second block of 3 x 3 conv
        bn2 = BatchNormalization(axis=chan_dim,
                                 epsilon=bn_eps,
                                 momentum=bn_mom)(conv1)
        act2 = Activation("relu")(bn2)

        # second block conv uses same padding and supplied strides
        conv2 = Conv2D(int(filters * 0.25),
                       kernel_size=(3, 3),
                       strides=strides,
                       padding="same",
                       use_bias=False,
                       kernel_regularizer=l2(reg))(act2)
        # %% ##################################################################
        # thrid block again valid padding
        bn3 = BatchNormalization(axis=chan_dim,
                                 epsilon=bn_eps,
                                 momentum=bn_mom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(filters,
                       kernel_size=(1, 1),
                       use_bias=False,
                       kernel_regularizer=l2(reg))(act3)
        # %% ##################################################################
        if red:
            shortcut = Conv2D(filters,
                              kernel_size=(1, 1),
                              strides=strides,
                              use_bias=False,
                              kernel_regularizer=l2(reg))(act1)
        x = add([conv3, shortcut])
        return x
        # %% ##################################################################

    @staticmethod
    def build(width, height, depth, classes, stages=[], filters=[],
              reg=0.0001, bn_eps=2e-5, bn_mom=0.9, dataset="cifar10"):
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = depth, height, width
            chan_dim = 1

        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chan_dim,
                               epsilon=bn_eps,
                               momentum=bn_mom)(inputs)
        if dataset == "cifar10":
            x = Conv2D(filters[0],
                       kernel_size=(3, 3),
                       use_bias=False,
                       padding="same",
                       kernel_regularizer=l2(reg))(x)
        # %% ##################################################################
        # loop over no of stages
        # stages = no. of residula modules to stack with K = from filters
        for i in range(len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = Resnet.residual_module(input_activation=x,
                                       filters=filters[i + 1],
                                       strides=stride,
                                       chan_dim=chan_dim,
                                       red=True,
                                       bn_eps=bn_eps,
                                       bn_mom=bn_mom)
            for j in range(stages[i] - 1):
                x = Resnet.residual_module(input_activation=x,
                                           filters=filters[i + 1],
                                           strides=(1, 1),
                                           chan_dim=chan_dim,
                                           red=False,
                                           bn_eps=bn_eps,
                                           bn_mom=bn_mom)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chan_dim,
                               epsilon=bn_eps,
                               momentum=bn_mom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="resnet")

        # return the constructed network architecture
        return model
