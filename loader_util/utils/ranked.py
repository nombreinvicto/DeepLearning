# import the necessary packages
import numpy as np


def rank5_accuracy(preds, labels):
    # init the vars
    rank1 = 0
    rank5 = 0

    # loop over the preds and ground truth labels
    for p, gt in zip(preds, labels):
        sort_indices = np.argsort(p)[::-1]

        if gt in sort_indices[:5]:
            rank5 += 1

        if gt == sort_indices[0]:
            rank1 += 1

    return rank1 / float(len(preds)), \
           rank5 / float(len(preds))
