"""
Multivariate linear regression
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Just simple linear regression with regularization - nothing new here
"""

from scipy import randn
from scipy import dot
from scipy import shape
from scipy import reshape
from scipy import size
from scipy.sparse import eye
from scipy.linalg import pinv

def ideal_data(num, dimX, dimY, _rank=None, noise=1):
    """Full rank data"""
    X = randn(num, dimX)
    W = randn(dimX, dimY)
    Y = dot(X, W) + randn(num, dimY) * noise
    return X, Y


class MultivariateRegressor(object):
    """
    Multivariate Linear Regressor.
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, X, Y, reg=None):
        if size(shape(X)) is 1:
            X = reshape(X, (-1, 1))
        if size(shape(Y)) is 1:
            Y = reshape(Y, (-1, 1))
        if reg is None:
            reg = 0

        W1 = pinv(dot(X.T, X) + reg * eye(size(X, 1)))
        W2 = dot(X, W1)
        self.W = dot(Y.T, W2)

    def __str__(self):
        return 'Multivariate Linear Regression'

    def predict(self, X):
        if size(shape(X)) == 1:
            X = reshape(X, (-1, 1))
        return dot(X, self.W.T)
