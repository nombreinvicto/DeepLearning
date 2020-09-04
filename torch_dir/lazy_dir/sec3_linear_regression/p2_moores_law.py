import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes
from sklearn.preprocessing import StandardScaler

sns.set()
# %%
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# %%
# load the data
data = pd.read_csv(filepath_or_buffer='moore.csv',
                   header=None).values
data.shape
# %%
# get the raw x's and y's
x = data[:, 0]
y = data[:, 1]
# %%

f, ax = plt.subplots(1, 1)  # type: Figure, axes.Axes
sns.scatterplot(x=x, y=y, ax=ax)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Y versus X")
plt.show()
# %%

# preprocess data
# logify the y
Y = np.log(y)

# now plot
f, ax = plt.subplots(1, 1)  # type: Figure, axes.Axes
sns.scatterplot(x=x, y=Y, ax=ax)
ax.set_xlabel("t")
ax.set_ylabel("log C")
ax.set_title("LogC versus t")
plt.show()
# %%

# lets construct a df
log_df = pd.DataFrame(data={'logC': Y, 't': x})
scaler = StandardScaler()
transformed_data = scaler.fit_transform(log_df)
transformed_log_df = pd.DataFrame({'logC_trans': transformed_data[:, 0],
                                   't_trans': transformed_data[:, 1]})
transformed_log_df.head()
# %%
# now plot the transformed data

f, ax = plt.subplots(1, 1)  # type: Figure, axes.Axes
sns.scatterplot(data=transformed_log_df,
                y='logC_trans', x='t_trans', ax=ax)
plt.show()
# %%
# convert datatype for torch
transformed_log_df["logC_trans"] = \
    transformed_log_df["logC_trans"].astype("float32")

transformed_log_df["t_trans"] = \
    transformed_log_df["t_trans"].astype("float32")

# also reshape for pytorch - needs 2d matrix
X = np.array(transformed_log_df["logC_trans"]).reshape(-1, 1)
Y = np.array(transformed_log_df["t_trans"]).reshape(-1, 1)
# %%

# model creation
model = nn.Linear(1, 1)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.7)

# define inputs and targets - tensorify them
inputs = torch.from_numpy(X)
targets = torch.from_numpy(Y)

# %%

# start the training process
n_epochs = 100
losses = []

for epoch in range(n_epochs):
    # zerp grad the optimizer
    optimizer.zero_grad()

    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # record the losses
    losses.append(loss.item())

    # backpass
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch + 1}/{n_epochs}, Loss: {loss.item(): 0.4f}")
# %%

# plot the loss trends
f, ax = plt.subplots(1, 1)  # type: Figure, axes.Axes
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Loss Vs Epochs")
sns.lineplot(x=range(1, n_epochs + 1),
             y=losses, ax=ax)
plt.show()
# %%

# plot the actual values and predicted line
f, ax = plt.subplots()  # type: Figure, axes.Axes

preds = model(inputs).detach().numpy().squeeze()
ax.set_xlabel("t_trans")
ax.set_ylabel("logC_trans")
sns.scatterplot(x=X.squeeze(),
                y=Y.squeeze(),
                label="Actual Values",
                ax=ax)
sns.lineplot(x=X.squeeze(),
             y=preds,
             label=" Predicted Line",
             ax=ax,
             color="orange")
plt.show()
# %%
