import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# %% ##################################################################
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=output_size))

    def forward(self, x):
        logits = self.model(x)
        return logits
# %% ##################################################################
