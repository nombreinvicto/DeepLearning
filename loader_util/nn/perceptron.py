# %%
# do the necessary imports
import numpy as np


# create the perceptreon class
class Perceptron:
    def __init__(self, N, alpha=0.1):
        # init the weight matrix and store the learning rate
        self.W = np.random.randn(N + 1)
        self.alpha = alpha

    def step(self, x):
        # apply the step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # insert a column of 1's as the last entry in the feature
        # matrix. this allows to treat the bias as a trainable param
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in range(epochs + 1):
            for x, target in zip(X, y):
                # calculate the prediction
                p = self.step(x.dot(self.W))

                # only perform weight update if our prediction
                # doesnt match the target value
                if p != target:
                    error = p - target

                    # update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        # ensure the input is a matrix
        X = np.atleast_2d(X)

        # check to see if the bias column shud be added or not
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(X.dot(self.W))
