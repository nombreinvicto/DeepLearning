# import the necessary packages

from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, \
    BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# %%

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        # initialise the model
        model = Sequential()
        input_shape = height, width, depth
        channel_dim = -1

        # if we are using channel first then update the input shape\
        if K.image_data_format() == "channel_first":
            input_shape = depth, height, width
            channel_dim = 1

        # first ABPD block
        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                         input_shape=input_shape,
                         padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # second ABPD block
        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
                         padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # third block
        model.add(Conv2D(384, (3, 3), padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(384, (3, 3), padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(Conv2D(256, (3, 3), padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # fourth FC blocks
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        # block 5 second Fc
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation('softmax'))

        return model
