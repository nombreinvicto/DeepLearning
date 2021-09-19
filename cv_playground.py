import torch
from loader_util.nn.conv import LeNetTorch
from torchsummary import summary

model = LeNetTorch(classes=50)
summary(model, input_size=(3, 224, 224))

import numpy as np
# %%
from functools import reduce
def apply_to_layer(layer):
    if isinstance(layer, torch.nn.Conv2d):
        n = reduce(lambda x, y: x * y, layer.weight.shape)
    elif isinstance(layer, torch.nn.Linear):
        n = layer.in_features
    else:
        return
    y = 1.0 / (np.sqrt(n))
    torch.nn.init.normal_(layer.weight, mean=0, std=y)


# %%
model.apply(apply_to_layer)
