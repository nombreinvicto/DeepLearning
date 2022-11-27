# import the required packages
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2


# %% ##################################################################
class AlexNet:
    @staticmethod
    def build(width,
              height,
              depth,
              classes,
              reg=0.0002):
        # initialise the model
        model = Sequential()
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = depth, height, width
            chan_dim = 1

        # block 1 : CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(filters=96,
                         kernel_size=(11, 11),
                         strides=(4, 4),
                         input_shape=input_shape,
                         padding="same",
                         kernel_regularizer=l2(reg)
                         )
                  )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        # output = 57 x 57 x 96
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
        # output = 28 x 28 x 96
        # %% ##################################################################
        # block 2 : CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(filters=256,
                         kernel_size=(5, 5),
                         padding="same",
                         kernel_regularizer=l2(reg)
                         )
                  )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        # output = 28 x 28 x 256
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
        # output = 13 x 13 x 256
        # %% ##################################################################
        # block 3(a) : CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(filters=384,
                         kernel_size=(3, 3),
                         padding="same",
                         kernel_regularizer=l2(reg)
                         )
                  )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        # output = 13 x 13 x 384
        # %% ##################################################################
        # block 3(b)  : CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(filters=384,
                         kernel_size=(3, 3),
                         padding="same",
                         kernel_regularizer=l2(reg)
                         )
                  )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        # output = 13 x 13 x 384
        # %% ##################################################################
        # block 3(c)  : CONV => RELU => POOL => DROPOUT
        model.add(Conv2D(filters=256,
                         kernel_size=(3, 3),
                         padding="same",
                         kernel_regularizer=l2(reg)
                         )
                  )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        # output = 13 x 13 x 256
        # %% ##################################################################
        # block 3(d)
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
        # output = 6 x 6 x 256
        # %% ##################################################################
        # block 4:
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # %% ##################################################################
        # softmax
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))
        # %% ##################################################################
        return model


# %% ##################################################################
if __name__ == '__main__':
    model = AlexNet().build(width=227,
                            height=227,
                            depth=3,
                            classes=2)
    print(model.summary())
