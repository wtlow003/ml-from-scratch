import logging
import sys

import numpy as np


class LogisticRegression:
    def __init__(self, lr: float = 0.0001, n_iters: float = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y, verbose=False):
        """Fit Logistic Regression algorithm to training data, given features X and target y.

        Args:
            X ([np.array]): Training features.
            y ([np.array]): Target.
        """
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent to optimize the cross-entropy (log loss) cost function
        for i in range(self.n_iters):
            if verbose:
                print(f"Processing epochs {i}...")
            linear_model = np.dot(X, self.weights) + self.bias
            # approximation of y
            y_predicted = self._sigmoid(linear_model)

            # deriatve of the weights, theta
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # deriative of the bias
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # updating the weights with respect to the deriative of the weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1.0 if i > 0.5 else 0 for i in y_predicted]

        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
