# import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


# %% ##################################################################

class FCHeadNet:
    @staticmethod
    def build(base_model: Model,
              classes: int,
              dense_nodes: int) -> Model:
        # init the head model that will be placed on top
        # of the base model then add FC layer
        head_model = base_model.output
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(dense_nodes, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(classes, activation="softmax")(head_model)
        return head_model
