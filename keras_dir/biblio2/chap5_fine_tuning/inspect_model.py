import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes
sns.set()
from tensorflow.keras.applications import VGG16

#%%

# construct the argument parser
args_dict = {
    "include_top": -1
}

#%%

# load the vgg network
print(f"[INFO] load the VGG network.....")
model = VGG16(weights="imagenet", include_top=args_dict["include_top"] > 0)


#%%
print(f"Showing layers.....")
# now loop over layers of model
for layer_no, layer in enumerate(model.layers):
    print(f"[INFO] {layer_no}\t{layer.__class__.__name__}")

#%%