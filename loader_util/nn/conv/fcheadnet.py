from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from typing import List


class FCHeadNet:
    @staticmethod
    def build(base_model: Model,
              output_classes: int,
              dense_layer_nodes: List[int]) -> Model:

        # .output retrieves the output tensor of a layer
        head_model = Flatten(name="flatten")(base_model.output)

        # now iterate through the dense layer nodes
        for nodes in dense_layer_nodes:
            head_model = Dense(nodes, activation="relu")(head_model)
            head_model = Dropout(0.5)(head_model)

        # after going thru the Dense layers, add the final softmax layer
        head_model = Dense(output_classes, activation="softmax")(head_model)

        return head_model
