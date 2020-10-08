import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes
import os

sns.set()
# %%
# import pytorch specific utils
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchsummary import summary
from datetime import datetime
import hiddenlayer as hl

# %%
cwd = os.getcwd()
data_file = f"{cwd}//df_depressionlevel_median_imputed.csv"
depression_data = pd.read_csv(data_file)
depression_data.head()
# %%

x_features = depression_data.drop(axis=1,
                                  labels=['Patient-ID',
                                          'depression_score',
                                          'date', 'week_num',
                                          'depression_level'])

y_features = depression_data['depression_score']
# %%
x_features_logscaled = x_features.copy()

for col in x_features.columns:
    skew = x_features[col].skew()
    if np.abs(skew) > 1:
        # then scale the feature
        x_features_logscaled[col] = x_features[col].apply(lambda x: np.log(x))

x_features_logscaled.head()
# %%

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_features_transformed = scaler.fit_transform(x_features_logscaled)
# %%
y_features = np.array(y_features)
y_features.shape
# %%

X = torch.from_numpy(x_features_transformed)
Y = torch.from_numpy(y_features).view(y_features.shape[0], 1)
# %%
from sklearn.model_selection import train_test_split

trainx, testx, trainy, testy = train_test_split(X, Y, test_size=0.2,
                                                random_state=42)

# %%
eps = torch.tensor(1e-7, dtype=torch.float32)


# %%
def coeff_determination(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    return (1 - ss_res / (ss_tot + eps))


# %%
from collections import OrderedDict

all_layers = OrderedDict()


def create_model(first_layer_units=1024):
    # create the constant part of the model
    all_layers['l0'] = nn.Linear(in_features=X.shape[1],
                                 out_features=first_layer_units)
    all_layers['r0'] = nn.ReLU()
    all_layers['b0'] = nn.BatchNorm1d(num_features=first_layer_units)

    # populate the neuron list
    units_list = []
    units_list.append(first_layer_units)
    while first_layer_units % 2 == 0:
        first_layer_units /= 2
        units_list.append(first_layer_units)

    # create the variable part of the model
    for i, units in enumerate(units_list):
        if i < len(units_list) - 1:
            all_layers[f"l{i + 1}"] = nn.Linear(in_features=units_list[i],
                                                out_features=units_list[i + 1])
            all_layers[f"r{i + 1}"] = nn.ReLU()
            all_layers[f"b{i + 1}"] = nn.BatchNorm1d(
                num_features=units_list[i + 1])

    # return the created model
    all_layers[f"l{i + 1}"] = nn.Linear(in_features=units_list[i + 1],
                                        out_features=1)

    # return the model
    return nn.Sequential(all_layers)
#%%
model = create_model(first_layer_units=16)
#%%