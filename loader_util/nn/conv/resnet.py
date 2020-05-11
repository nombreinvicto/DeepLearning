# import the required packages
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    AveragePooling2D, MaxPooling2D, ZeroPadding2D, Activation, Dense, \
    Flatten, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# %%

class ResNet:
    @staticmethod
    def residual_module(data,
                        K,
                        stride,
                        chanDim,
                        red=False,
                        reg=0.0001,
                        bnEps=2e-5,
                        bnMom=0.9):
        # shortcut branch of resnet init as input
        shortcut = data

        # first block of resnet module is 1x1 conv
        bn1 = BatchNormalization(axis=chanDim,
                                 epsilon=bnEps,
                                 momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(filters=int(K * 0.25),
                       kernel_size=(1, 1),
                       use_bias=False,
                       kernel_regularizer=l2(reg))(act1)

        # second block of 3x3 convs
        bn2 = BatchNormalization(axis=chanDim,
                                 epsilon=bnEps,
                                 momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(filters=int(K * 0.25),
                       kernel_size=(3, 3),
                       strides=stride,
                       padding="same",
                       use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # third block of 1x1 convs
        bn3 = BatchNormalization(axis=chanDim,
                                 epsilon=bnEps,
                                 momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(filters=int(K),
                       kernel_size=(1, 1),
                       use_bias=False,
                       kernel_regularizer=l2(reg))(act3)

        # check if we need to reduce spatial dims, to eliminate need of max
        # pool
        if red:
            shortcut = Conv2D(K,
                              (1, 1),
                              strides=stride,
                              use_bias=False,
                              kernel_regularizer=l2(reg))(act1)

        # add together shortcut amd final conv
        x = add([conv3, shortcut])

        # return tensor
        return x

    @staticmethod
    def build(width,
              height,
              depth,
              classes,
              stages,
              filters,
              reg=0.0001,
              bnEps=2e-5,
              bnMom=0.9,
              dataset="cifar"):
        # init the input shape parameter
        inputShape = height, width, depth
        chanDim = -1

        # if using channel last update the param
        if K.image_data_format() == "channel_first":
            inputShape = depth, height, width
            chanDim = 1

        # set the input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim,
                               epsilon=bnEps,
                               momentum=bnMom)(inputs)

        # check of utilising cifar10
        if dataset == "cifar":
            # apply single CONV layer
            x = Conv2D(filters[0], (3, 3),
                       use_bias=False,
                       padding="same",
                       kernel_regularizer=l2(reg))(x)

        # loop over the no of stages
        for i in range(len(stages)):
            # init the stride and then apply a residual module used to
            # reduce spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x,
                                       filters[i + 1],
                                       stride,
                                       chanDim,
                                       red=True,
                                       bnEps=bnEps,
                                       bnMom=bnMom)

            # loop over the no of layers in the stage
            for j in range(stages[i] - 1):
                # apply a RESNET module
                x = ResNet.residual_module(x,
                                           K=filters[i + 1],
                                           stride=(1, 1),
                                           chanDim=chanDim,
                                           bnEps=bnEps,
                                           bnMom=bnMom)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D(pool_size=(8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation('softmax')(x)

        # create the model
        model = Model(inputs, x, name="resnet")

        return model



