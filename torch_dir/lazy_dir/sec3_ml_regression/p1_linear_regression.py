import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes

sns.set()
# %%
import torch
from torch import nn
# %%

# generate synthetic data
N = 20  # 20 datapoints
x = np.random.random(N) * 10 - 5

# a line plus some noise
y = 0.5 * x - 1 + np.random.randn(N)

f, ax = plt.subplots(1, 1)  # type: axes.Axes, axes.Axes
sns.scatterplot(x=x, y=y, ax=ax)
ax.set_title("Y versus X")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()
# %%

# create the linear regression model
model = nn.Linear(in_features=1,
                  out_features=1,
                  bias=True)

# loass and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# %%

# reshape datasets
x = x.reshape(N, -1)
y = y.reshape(N, -1)

inputs = torch.from_numpy(x.astype('float32'))
targets = torch.from_numpy(y.astype('float32'))
type(inputs)
# %%

# train the model
n_epochs = 30
losses = []

for epoch in range(n_epochs):
    # zero the optimizer
    optimizer.zero_grad()

    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # keep the loss so that we can plot it later
    losses.append(loss.item())

    # backward optimize
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch + 1}/{n_epochs}, Loss: {loss.item(): 0.4f}")
# %%

# plot the losses versus epochs

f, ax = plt.subplots(1, 1)
x_plot = list(range(1, n_epochs + 1))
sns.lineplot(x=x_plot, y=losses)
ax.set_xlabel("Loss")
ax.set_ylabel("Epochs")
plt.show()
# %%

# plot the actual and predicted values
f, ax = plt.subplots(1, 1)  # type: axes.Axes
x_plot = x.squeeze()
y_plot = y = y.squeeze()
preds = model(inputs).detach().numpy().squeeze()
sns.scatterplot(x=x_plot, y=y_plot, ax=ax, label="Actual Values")
sns.lineplot(x=x_plot, y=preds, ax=ax, label="Predicted Line", color="red")
ax.set_ylabel("Y")
ax.set_xlabel("X")
ax.set_title("Y versus X")
plt.show()
# %%
