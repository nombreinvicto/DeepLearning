import sys, os, json
import numpy as np
import pandas as pd
import seaborn as sns
import argparse, progressbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from typing import List

sns.set()
# %%
f: Figure
ax: Axes
f, ax = plt.subplots(1, 1, figsize=(16, 8))
# %%

# scaled version of vector
colors = ["red", "green", "blue"]
v = np.array([1, 2])
sns.lineplot(x=[0, v[0]], y=[0, v[1]], color="red", ax=ax, label="main")
for i in range(10):
    s = np.random.randn() # Return a sample (or samples) from the "standard normal" distribution. u = 0, stddev = 1
    new_v = s * v
    sns.lineplot(x=[0, new_v[0]], y=[0, new_v[1]], color=np.random.choice(colors), ax=ax, label=f"{i}", linewidth=2.5)
plt.show()
#%%

