from torch.utils.data import Dataset
from typing import List
import numpy as np
from PIL import Image
import os


class CustomTorchDataset(Dataset):
    def __init__(self, pathList: List, tranforms):
        super(CustomTorchDataset, self).__init__()
        if not isinstance(pathList, list):
            raise TypeError("Supplied pathList is not a valid list im image paths")
        self.pathList = np.random.permutation(pathList)
        self.classes = sorted(list(np.unique([d.split(os.path.sep)[-2] for d in self.pathList])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.transforms = tranforms

    def __len__(self):
        return len(self.pathList)

    def __getitem__(self, idx):
        path = self.pathList[idx]
        target = path.split(os.path.sep)[-2]
        image = Image.open(path)
        if self.transforms:
            image = self.transforms(image)

        return image, self.class_to_idx[target]

    def _get_unique_classes(self):
        return
