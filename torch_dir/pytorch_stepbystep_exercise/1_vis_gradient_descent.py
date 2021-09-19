import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes

sns.set()

# %%
true_b = 1
true_w = 2
N = 100
# data generation
np.random.seed(42)
x = np.random.randn(N, 1)
epsilon = 0.1 * (np.random.randn(N, 1))
y = true_b + true_w * x + epsilon
# %%
# shuffle indices
idx = np.arange(N)
np.random.shuffle(idx)
# %%

# use first 80 random indices for train
train_idx = idx[:int(N * 0.8)]
val_idx = idx[int(N * 0.8):]

# generate xtrain and valid
x_train, y_train = x[train_idx], y[train_idx]
x_valid, y_valid = x[val_idx], y[val_idx]
# %%
f: Figure
ax: axes
f, ax = plt.subplots(1, 2, figsize=(15, 7))
sns.scatterplot(x=x_train.flatten(), y=y_train.flatten(), ax=ax[0])
sns.scatterplot(x=x_valid.flatten(), y=y_valid.flatten(), ax=ax[1])
plt.show()
# %%
# random init the params
np.random.seed(42)
b = np.random.randn(1)
w = np.random.rand(1)
print(b, w)
# %%

# compute models predictions = forward pass
yhat = b + w * x_train
# %%

# compute the error
error = yhat - y_train
loss = (error ** 2).mean()
print(f"[INFO] loss: {loss}......")
# %%
b_range = np.linspace(true_b - 3, true_b + 3, 101)
w_range = np.linspace(true_w - 3, true_w + 3, 101)
bs, ws = np.meshgrid(b_range, w_range)
print(f"[INFO] {bs.shape} and {ws.shape}......")
#%%
