# import the required packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, \
    Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K


class MinVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialise the model with input shape assuming channel last
        model = Sequential()
        inputShape = height, width, depth
        chanDim = -1

        # if using channel first then update inpute shape
        if K.image_data_format() == 'channel_first':
            inputShape = depth, height, width
            chanDim = 1

        # now construct the architecture
        model.add(Conv2D(filters=32,
                         kernel_size=(3, 3),
                         padding='same',
                         input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # add next layer
        model.add(Conv2D(filters=64,
                         kernel_size=(3, 3),
                         padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first and only set of FC layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
