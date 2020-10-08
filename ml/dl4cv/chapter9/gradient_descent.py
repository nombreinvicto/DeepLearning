# do the imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def predict(X: np.ndarray, W: np.ndarray):
    # take the dot product between features and weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply a step function to threshold the outputs to binary class
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
                help=" # of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
args = vars(ap.parse_args())

# generate a 2 class classificatin problem with 1,000 data points
# where each datapoint is a 2D feature vector
X, y = make_blobs(n_samples=1000, n_features=2, centers=2,
                  cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
# insert a column of 1's as the last entry in the feature matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of 
# the data for training and the remaining 50% for testing

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.5,
                                                random_state=42)
# initialise our weight matrix with random values
print(f"[INFO] training.....")

W = np.random.randn(X.shape[1], 1)
losses = []

# # loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    # trainX.W will be a 500x1 vector of scores passed to the sigmoid
    # a 500x1 preds probability vector is then returned
    preds = sigmoid_activation(trainX.dot(W))

    # error is the difference between the 500x1 probability vector
    # and the 500x1 true trainY vector
    error = preds - trainY

    # aggregate all losses as square loss into a single  number
    loss = np.sum(error ** 2)
    losses.append(loss)

    d = error * sigmoid_deriv(preds)
    gradient = trainX.T.dot(d)

    # now male the gradient descent update of the weight matrix
    W += -args['alpha'] * gradient

    # display loss info after every 5 epochs
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f'[INFO] epoch={epoch}, loss={loss}')

# evaluate our model
print(f'[INFO] Evaluating model .......')
testPreds = predict(testX, W)
print(classification_report(testY, testPreds))
