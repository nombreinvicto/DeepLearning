# import the required packages
import numpy as np


# %% ##################################################################

def rank5_accuracy(preds: np.ndarray, labels: np.ndarray):
    # init the rank1 and rank5 accuracies
    rank1 = 0
    rank5 = 0

    for pred, label in zip(preds, labels):
        pred: np.ndarray
        label: int

        high_to_low_prob_indices = np.argsort(pred)[::-1]

        if high_to_low_prob_indices[0] == label:
            rank1 += 1

        if label in high_to_low_prob_indices[:5]:
            rank5 += 1

    rank1_acc = rank1 / (len(preds))
    rank5_acc = rank5 / (len(preds))
    return rank1_acc, rank5_acc
