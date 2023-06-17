# import the necessary packages
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# %% ##################################################################
def make_pairs(images: np.ndarray,
               labels: np.ndarray):
    # to hold (image, image) pairs
    pair_images = []

    # to hold corr labels (same=1 or not=0)
    pair_labels = []

    # get unique no of classes
    num_classes = len(np.unique(labels))

    # idx is a list of np arrays, each array
    # containing indices of where labels == i, where i starts from 0
    idx = [np.where(labels == i)[0] for i in range(0, num_classes)]

    # loop over the array of images supplied
    print(f"[INFO] looping over: {len(images)} times......")
    for idxa in range(len(images)):
        current_image = images[idxa]
        current_label = labels[idxa]

        # randomly pick an image belonging to the same class label
        idxb = np.random.choice(idx[current_label])
        pos_image = images[idxb]

        pair_images.append([current_image, pos_image])
        pair_labels.append([1])

        # randomly choose a negative image for the current image
        neg_idx = np.where(labels != current_label)[0]
        neg_image = images[np.random.choice(neg_idx)]

        pair_images.append([current_image, neg_image])
        pair_labels.append([0])

    # return the pairs after iterating thru all images
    return np.array(pair_images), np.array(pair_labels)


# %% ##################################################################
def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors

    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)

    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


# %% ##################################################################
def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)

    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squared_preds = K.square(preds)
    squared_margin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squared_preds + (1 - y) * squared_margin)

    # return the computed contrastive loss to the calling function
    return loss


# %% ##################################################################
def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()

    if H.history.get("loss"):
        plt.plot(H.history["loss"], label="train_loss")

    if H.history.get("val_loss"):
        plt.plot(H.history["val_loss"], label="val_loss")

    if H.history.get("accuracy"):
        plt.plot(H.history["accuracy"], label="train_acc")

    if H.history.get("val_accuracy"):
        plt.plot(H.history["val_accuracy"], label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)
