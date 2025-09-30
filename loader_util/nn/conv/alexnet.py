from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Dense,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# %% ##################################################################
class AlexNet():
    @staticmethod
    def build(height, width, depth, classes, reg=0.0002):
        model = Sequential()
        input_shape = height, width, depth
        # i.e the last dim is chan dim for channel last configuration
        chan_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = depth, height, width
            chan_dim = 1

        # build the model
        # block -1
        model.add(Conv2D(filters=96,
                         kernel_size=(11, 11),
                         strides=(4, 4),
                         padding="same",
                         input_shape=input_shape,
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # block 2 ABPD
        model.add(Conv2D(filters=256,
                         kernel_size=(5, 5),
                         padding="same",
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
        # %% ##################################################################
        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        model.add(Conv2D(384, (3, 3), padding="same",
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(384, (3, 3), padding="same",
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(256, (3, 3), padding="same",
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
        # %% ##################################################################
        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Block #5: second set of FC => RELU layers
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # %% ##################################################################
        # softmax classifier
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        return model
        # %% ##################################################################


if __name__ == '__main__':
    alexnet = AlexNet.build(227, 227, 3, classes=10, reg=0.0002)
    print(alexnet.summary())
