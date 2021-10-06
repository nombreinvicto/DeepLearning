# %%
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, \
    BatchNormalization, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


# %%

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chan_dim, red=False, reg=0.0001, bn_eps=2e-5, bn_mom=0.9):
        # shortcut or identity branch
        shortcut = data
        # first block = BN + ReLU + Conv
        bn1 = BatchNormalization(axis=chan_dim, momentum=bn_mom, epsilon=bn_eps)(data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(filters=(K // 4), kernel_size=(1, 1), strides=(1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # second block BN + ReLU + Conv
        bn2 = BatchNormalization(axis=chan_dim, momentum=bn_mom, epsilon=bn_eps)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(filters=(K // 4), kernel_size=(3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

        # third block
        bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(filters=K, kernel_size=(1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if red:
            shortcut = Conv2D(filters=K, kernel_size=(1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(shortcut)

        return add([conv3, shortcut])

    @staticmethod
    def build(width, height, depth, classes, stages=(), filters=(), reg=0.0001, bn_eps=2e-5, bn_mom=0.9, dataset="cifar"):
        # init the input shape
        input_shape = height, width, depth
        chan_dim = -1

        if K.image_data_format() == "channel_first":
            input_shape = depth, height, width
            chan_dim = 1

        # set input
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(inputs)

        if dataset == "cifar":
            x = Conv2D(filters=filters[0], kernel_size=(3, 3), use_bias=False, padding='same', kernel_regularizer=l2(reg))(x)

        # loop over the stages
        for i in range(len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(data=x, K=filters[i + 1], stride=stride, chan_dim=chan_dim, red=True, bn_eps=bn_eps, bn_mom=bn_mom)

            # loop  over the number of j stages for each stage
            for j in range(stages[i] - 1):
                # create more resnet modules without reduction
                x = ResNet.residual_module(data=x, K=filters[i + 1], stride=(1, 1), chan_dim=chan_dim, bn_eps=bn_eps, bn_mom=bn_mom)

        # apply BN -> ACT -> POOL
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(8, 8))(x)

        # softmax
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation('softmax')(x)

        # create model
        model = Model(inputs=inputs, outputs=x, name="resnet")
        return model


if __name__ == '__main__':
    my_model = ResNet.build(width=32, height=32, depth=3, classes=10, stages=(3, 4, 6), filters=(64, 128, 256, 512))
    print(my_model.summary())
# %%
