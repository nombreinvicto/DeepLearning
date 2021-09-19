import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse, progressbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes

sns.set()
# %%

train_images_path = r"C:\Users\mhasa\Google Drive\Tutorial " \
                    r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\tiny-imagenet-200\tiny-imagenet-200\train"
valid_images_path = r"C:\Users\mhasa\Google Drive\Tutorial " \
                    r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\tiny-imagenet-200\tiny-imagenet-200\val\images"
val_mapping = r"C:\Users\mhasa\Google Drive\Tutorial " \
    r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\tiny-imagenet-200\tiny-imagenet-200\val\val_annotations.txt"
wordnet_ids = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\tiny-imagenet-200\tiny-imagenet-200\wnids.txt"
word_labels = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\tiny-imagenet-200\tiny-imagenet-200\words.txt"
hdf5_base = r"C:\Users\mhasa\Google Drive\Tutorial " \
            r"Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\tiny-imagenet-200\hdf5"
output_base = r"C:\Users\mhasa\Google Drive\Tutorial Corner\PYTH\DeepLearning\DeepLearning-DL4CV\ImageDatasets\tiny-imagenet-200\output"

train_hdf5 = f"{hdf5_base}//tiny_imagent_train.hdf5"
valid_hdf5 = f"{hdf5_base}//tiny_imagent_valid.hdf5"
test_hdf5 = f"{hdf5_base}//tiny_imagent_test.hdf5"
dataset_mean = f"{output_base}//tiny_imagenet_rgb_mean.json"
model_path = f"{output_base}//saved_model.hdf5"
fig_path = f"{output_base}//tiny_imagenet_performance.png"
json_path = f"{output_base}//tine_imagenet_history.json"

# %%
num_classes = 200
num_test_images = 50 * num_classes
