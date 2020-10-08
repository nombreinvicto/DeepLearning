from sklearn.metrics import confusion_matrix

def test_model(test_labels, pred_labels):
    tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels, labels=[0, 1]).ravel()
    accuracy = float((tp + tn)) / (tp + tn + fp + fn)
    recall = float(tp) / (tp + fn)
    precision = float(tp) / (tp + fp)
    F1 = 2 * precision * recall / (precision + recall)

    return tn, fp, fn, tp, accuracy