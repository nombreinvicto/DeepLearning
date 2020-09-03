import numpy as np


def rank5_accuracy(preds: np.ndarray, labels: np.ndarray):
    # initialise the rank 1 and rank 5 accuracies
    rank1 = 0
    rank5 = 0

    # loop over predictions and ground truth labels
    for p, gt in zip(preds, labels):
        # sort the probabilities by their index in descending order so that
        # the more confident guesses are at the front of the list
        p = np.argsort(p)[::-1]

        # check if ground truth is the top 5 preds
        if gt in p[:5]:
            rank5 += 1

        # check to see if the ground truth is the #1 prediction
        if gt == p[0]:
            rank1 += 1

    # compute the final rank accuracies
    rank1 /= float(len(preds))
    rank5 /= float(len(preds))

    # return a tuple of the ran accuracies
    return rank1, rank5
