import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


# Following functions can be placed in class LogisticModel

def sigmoid(x):
    """
    Numerically stable version of sigmoid (no overflow warnings).
    :param x: array of arguments
    :return: sigmoid of x
    """
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def predict_probabilities(beta, x):
    z = np.dot(x, beta)
    return np.apply_along_axis(sigmoid, axis=1, arr=z)


def cost_function(beta, x, y):
    """
    Cross-entropy/log-likelihood cost function (for GD and SGD)
    :param beta:
    :param x:
    :param y:
    :return:
    """
    m = x.shape[0]
    h = predict_probabilities(beta, x)
    return -(1/m)*np.sum(y*np.log(h) + (1-y)*np.log(1-h))


def accuracy(y: np.array, prediction: np.array) -> float:
    """
    Given two vectors consisting of 0 and 1, returns accuracy.
    :param y: true values (0 and 1)
    :param prediction: prediciton (0 and 1)
    :return: accuracy
    """
    return np.mean(y == prediction)


def precision(y: np.array, prediction: np.array) -> float:
    """
    Given two vectors consisting of 0 and 1, returns precision.
    We assume that 1 represents positive value.
    :param y: true values (0 and 1)
    :param prediction: prediciton (0 and 1)
    :return: precision
    """
    true_positives = np.sum((y == 1) & (prediction == 1))
    positives = np.sum(prediction)
    return 0 if positives == 0 else true_positives / positives


def recall(y: np.array, prediction: np.array) -> float:
    """
    Given two vectors consisting of 0 and 1, returns recall.
    We assume that 1 represents positive value.
    :param y: true values (0 and 1)
    :param prediction: prediciton (0 and 1)
    :return: recall
    """
    true_positives = np.sum((y == 1) & (prediction == 1))
    true_values = np.sum(y)
    return 0 if true_values == 0 else true_positives / true_values


def F_measure(y: np.array, prediction: np.array) -> float:
    """
    Given two vectors consisting of 0 and 1, returns the F measure.
    We assume that 1 represents positive value.
    :param y: true values (0 and 1)
    :param prediction: prediciton (0 and 1)
    :return: recall
    """
    r = recall(y, prediction)
    p = precision(y, prediction)
    return 0 if r + p == 0 else 2 * r * p / (r + p)


class LogisticModel(object):
    """

    """
    def __init__(self, var_names, X, y, opt_alg):
        self.var_names = var_names
        self.X = np.c_[np.ones((X.shape[0], 1)), X]  # adding bias
        self.y = y
        self.opt_alg = opt_alg
        self.weights = np.ones(self.X.shape[1]).reshape(self.X.shape[1], 1)

    def fit(self, n_epochs):
        ...

    def IRLS(self, n_epochs, eps):
        # IN PROGRESS...
        # based on chapter 4.3.3 from Bishop - Pattern Recognition And Machine Learning
        y = self.y.reshape(self.y.size, 1)
        prev_delta = np.zeros(self.X.shape[1]).reshape(self.X.shape[1], 1)
        for i in range(n_epochs):
            y_ = predict_probabilities(self.weights, self.X)
            R = np.identity(y_.size)
            R = R * (y_ * (1 - y_))
            delta = np.linalg.inv(self.X.T.dot(R).dot(self.X)).dot(self.X.T.dot(y_ - y))
            if np.linalg.norm(delta - prev_delta) < eps:
                # maybe some info about number of iterations
                break
            self.weights -= delta
            prev_delta = delta

    def GD(self, n_epochs, learning_rate):
        pass

    def SGD(self, n_epochs, learning_rate):
        pass
