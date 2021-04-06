import numpy as np
import pandas as pd

# Following functions can be placed in class LogisticModel


def sigmoid(x):
    """
    Numerically stable version of sigmoid (no overflow warnings).
    :param x: array of arguments
    :return: sigmoid of x
    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def predict_probabilities(beta, x):
    z = np.dot(x, beta).reshape(x.shape[0], 1)
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
    return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


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


class LogisticModel:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, opt_alg=None):
        self.var_names = X.columns
        self.X = np.c_[np.ones((X.shape[0], 1)), np.array(X)]  # adding bias
        self.y = np.array(y)
        self.opt_alg = opt_alg
        self.weights = np.zeros((self.X.shape[1], 1))

    def fit(self, X: pd.DataFrame, return_probabilities: bool = False) -> pd.DataFrame:
        """
        Returns class to which t
        :param X: input dataframe
        :param return_probabilities: should the model return probabilities of belonging to class 1?
        :return: vector of 1 and 0, which represents predicted classes
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        if return_probabilities:
            return predict_probabilities(self.weights, X)
        else:
            return np.apply_along_axis(
                np.round, arr=predict_probabilities(self.weights, X), axis=1
            )

    def IRLS(self, n_epochs=100, eps=1e-06, print_progress=False):
        # based on chapter 4.3.3 from Bishop - Pattern Recognition And Machine Learning
        prev_delta = np.zeros(self.X.shape[1]).reshape(self.X.shape[1], 1)
        y = self.y.reshape(self.y.size, 1)
        for i in range(n_epochs):
            y_ = predict_probabilities(self.weights, self.X)
            R = np.identity(y_.size)
            R = R * (y_ * (1 - y_))
            delta = np.linalg.inv(self.X.T.dot(R).dot(self.X)).dot(self.X.T.dot(y_ - y))
            self.weights -= delta

            if print_progress:
                print(f"Number of epoch: {i+1}/{n_epochs}")

            if np.linalg.norm(delta - prev_delta) < eps:
                print(f"Algorithm converged after {i+1} iterations")
                break
            prev_delta = delta

    def GD(self, n_epochs=100, learning_rate=0.05, eps=1e-06):
        prev_delta = np.zeros(self.X.shape[1]).reshape(self.X.shape[1], 1)
        for i in range(n_epochs):
            prediction = predict_probabilities(self.weights, self.X)
            delta = np.matmul(
                self.X.transpose(),
                (prediction - self.y) * prediction * (1 - prediction),
            )
            self.weights -= delta * learning_rate

            if np.linalg.norm(delta - prev_delta) < eps:
                print(f"Algorithm converged after {i+1} iterations")
                break
            prev_delta = delta

    def SGD(self, n_epochs=100, learning_rate=0.1, batch_size=1, random_state=None, eps=1e-6):
        # based on https://realpython.com/gradient-descent-algorithm-python/

        n_obs = self.X.shape[0]
        n_vars = self.X.shape[1]
        xy = np.c_[self.X.reshape(n_obs, -1), self.y.reshape(n_obs, 1)]

        # random number generator for permutation of observations
        seed = None if random_state is None else int(random_state)
        rng = np.random.default_rng(seed=seed)

        batch_size = int(batch_size)
        if not 0 < batch_size <= n_obs:
            raise ValueError(
                "Batch size must be greater than zero and not greater than"
                "the number of observations!"
            )

        for i in range(n_epochs):
            rng.shuffle(xy)
            batch_division = np.arange(batch_size, n_obs, batch_size)
            xy_batches = np.split(xy, batch_division)
            for n_batch in range(len(batch_division)):
                x_batch, y_batch = xy_batches[n_batch][:, :-1], xy_batches[n_batch][:, -1:]
                prediction = predict_probabilities(self.weights, x_batch)
                delta = np.matmul(
                    x_batch.transpose(),
                    (prediction - y_batch) * prediction * (1 - prediction),
                )

                self.weights -= delta * learning_rate
