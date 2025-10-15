import numpy as np


def rankn_accuracy(preds: np.ndarray,
                   labels: np.ndarray,
                   rank=5):
    # init the accuracies
    rank1_acc = 0
    rankn_acc = 0

    # preds expected to be nxd array of probability
    # which could come as output of model.predict()
    # labels expected to be (n,) array of labels/indices
    # make sure rankn conforms to probability dimension
    assert preds.shape[1] <= rank, \
        "predictions array should have a probability " \
        "dimension less than or equal to rank number"

    for prob_array, ground_truth in zip(preds, labels):
        # argsort send the index of the smallest element first
        # using reversed slicing to reverse that
        prob_array_indices = np.argsort(prob_array)[::-1]

        if ground_truth in prob_array_indices[:rank]:
            rankn_acc += 1

        if ground_truth == prob_array_indices[0]:
            rank1_acc += 1

    # after iterating thru all rows, return final accuracies
    rank1_acc /= preds.shape[0]
    rankn_acc /= preds.shape[0]

    return rank1_acc, rankn_acc
