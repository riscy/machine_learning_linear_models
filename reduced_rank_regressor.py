"""
Reduced rank regression class.
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Optimal linear 'bottlenecking' or 'multitask learning'.
"""

from scipy import randn
from scipy import dot
from scipy import shape
from scipy import reshape
from scipy import size
from scipy.sparse import eye
from scipy.linalg import pinv
from scipy.linalg import svd

def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X = randn(num, dimX)
    W = dot(randn(dimX, rrank), randn(rrank, dimY))
    Y = dot(X, W) + randn(num, dimY) * noise
    return X, Y


class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, X, Y, rank, reg=None):
        if size(shape(X)) == 1:
            X = reshape(X, (-1, 1))
        if size(shape(Y)) == 1:
            Y = reshape(Y, (-1, 1))
        if reg is None:
            reg = 0
        self.rank = rank

        CXX = dot(X.T, X) + reg * eye(size(X, 1))
        CXY = dot(X.T, Y)
        _U, _S, V = svd(dot(CXY.T, dot(pinv(CXX), CXY)))
        self.W = V[0:rank, :].T
        self.A = dot(pinv(CXX), dot(CXY, self.W)).T

    def __str__(self):
        return 'Reduced Rank Regressor (rank = {})'.format(self.rank)

    def predict(self, X):
        """Predict Y from X."""
        if size(shape(X)) == 1:
            X = reshape(X, (-1, 1))
        return dot(X, dot(self.A.T, self.W.T))
