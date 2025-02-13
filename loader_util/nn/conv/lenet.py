# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Flatten, \
    Dense
from tensorflow.keras import backend as K


# build the network
class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = height, width, depth

        # if we are using "channels first" then update the input shape
        if K.image_data_format() == 'channels_first':
            inputShape = depth, height, width

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(filters=20,
                         kernel_size=(5, 5),
                         padding='same',
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(filters=50,
                         kernel_size=(5, 5),
                         padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2),
                            strides=(2, 2)))

        # FC layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
