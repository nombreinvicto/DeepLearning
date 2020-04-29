# import the required packages
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model


class FCHeadNet:
    @staticmethod
    def builld(baseModel: Model, classes, D):
        # initialise the head model that will be placed on top of the base
        # then add FC layer
        headModel = baseModel.output
        print("BaseModel Out Shape: ",baseModel.output.shape)
        headModel = Flatten(name='flatten')(headModel)

        headModel = Dense(D[0], activation='relu')(headModel)
        headModel = Dropout(0.5)(headModel)

        headModel = Dense(D[1], activation='relu')(headModel)
        headModel = Dropout(0.5)(headModel)

        headModel = Dense(D[2], activation='relu')(headModel)
        headModel = Dropout(0.5)(headModel)

        # add a softmax
        headModel = Dense(classes, activation='softmax')(headModel)

        # return model
        return headModel
