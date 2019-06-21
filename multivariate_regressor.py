"""
Multivariate linear regression
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Just simple linear regression with regularization - nothing new here
"""

import numpy as np
from scipy import sparse


def ideal_data(num, dimX, dimY, _rank=None, noise=1):
    """Full rank data"""
    X = np.random.randn(num, dimX)
    W = np.random.randn(dimX, dimY)
    Y = np.dot(X, W) + np.random.randn(num, dimY) * noise
    return X, Y


class MultivariateRegressor(object):
    """
    Multivariate Linear Regressor.
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, X, Y, reg=None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0

        W1 = np.linalg.pinv(np.dot(X.T, X) + reg * sparse.eye(np.size(X, 1)))
        W2 = np.dot(X, W1)
        self.W = np.dot(Y.T, W2)

    def __str__(self):
        return 'Multivariate Linear Regression'

    def predict(self, X):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.dot(X, self.W.T)
