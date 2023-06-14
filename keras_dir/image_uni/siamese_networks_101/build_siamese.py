from tensorflow.keras.datasets import mnist
from imutils import build_montages
import numpy as np
import cv2
# %% ##################################################################
def make_pairs(images:np.ndarray, labels:np.ndarray):

    # to hold (image, image) pairs
    pair_images = []

    # to hold corr labels (same or not)
    # for mnist it shud be 0 - 9
    pair_labels = []

    # get unique no of classes
    num_classes = len(np.unique(labels))

