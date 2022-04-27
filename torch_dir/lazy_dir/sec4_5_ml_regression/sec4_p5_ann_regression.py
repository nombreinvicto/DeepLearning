#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

sns.set()

# In[2]:


import torch
from torch import nn
from torch import optim

# In[3]:


# make dataset
N = 1000
# random gives random floats in the half interval [0-1]
# x is 1000 x 2. First column is x1, second is x2
x = np.random.random((N, 2)) * 6 - 3  # shift interval to [-3, +3]

# In[4]:


# make the y
# we have 3 variables, x1, x2 and y hence its a 3d plot
y = np.cos(2 * x[:, 0]) + np.cos(3 * x[:, 1])

# In[5]:


# plot the graph
fig = plt.figure()  # type: Figure
ax = fig.add_subplot(111, projection="3d")  # type: Axes3D
ax.scatter(x[:, 0], x[:, 1], y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

# In[6]:


# build the model with inputs x1, x2 and output y
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=1)
)

# In[7]:


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# In[8]:


# train the model
xtrain = torch.from_numpy(x.astype(np.float32))
ytrain = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)

epochs = 1000
train_losses = np.zeros(epochs)

for i in range(epochs):
    # zero the optimizer
    optimizer.zero_grad()

    # forward pass
    outputs = model(xtrain)
    loss = criterion(outputs, ytrain)

    # backward step
    loss.backward()
    optimizer.step()

    train_losses[i] = loss.item()

    if (i + 1) % 10 == 0:
        print(f"[INFO] epoch: {i + 1} with train_loss: {loss.item()}......")

# In[11]:


f, ax = plt.subplots(1, 1, figsize=(12, 8))  # type: Figure, Axes
sns.lineplot(x=np.arange(epochs) + 1, y=train_losses, ax=ax, color="green", linewidth=2)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss");

# In[12]:


# plot the actual points in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, color="red")


# now plot the prediction surface
with torch.no_grad():
    line = np.linspace(-3, 3, 50)  # generate 50 data points in the range [-3, 3] remember our x1 and x2 are in that range
    x1, x2 = np.meshgrid(line, line)

    # x1 and x2 are as grids. need to convert them into data format the model expectsas N X 2 matrix
    x = np.vstack((x1.flatten(), x2.flatten())).T
    x_torch = torch.from_numpy(x.astype(np.float32))
    yhat = model(x_torch).numpy().flatten()
    ax.plot_trisurf(x[:, 0], x[:, 1], yhat, linewidth=0.2, antialiased=True)
    plt.show();
# In[5]:


# In[5]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
