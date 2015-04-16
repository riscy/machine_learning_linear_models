#
# Multivariate linear regression
# Requires pylab to be installed.
#
# Implemented by Chris Rayner (2015)
# dchrisrayner AT gmail DOT com
#
# Just simple linear regression with regularization -- nothing 'new' here.
#
# URL: http://www.cs.ualberta.ca/~rayner/
# Git: https://github.com/riscy/mllm/

from pylab import *

def idealData(num, dimX, dimY, noise=1):
    """
    Example of ideal data for this model.
    """

    X = randn(num, dimX)
    W = randn(dimX, dimY)
    Y = dot(X, W) + randn(num, dimY) * noise

    return X, Y;


class mllmLR:
    """
    Multivariate Linear Regression
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - reg is a regularization parameter (optional).
    """

    def __init__ (self, X, Y, reg=None):

        if size(shape(X)) is 1:
            X = reshape(X, (-1,1))
        if size(shape(Y)) is 1:
            Y = reshape(Y, (-1,1))

        if reg is None:
           reg = 0

        W1 = pinv(dot(X.T, X) + reg * eye(size(X, 1)))
        W2 = dot(X, W1)
        self.W = dot(Y.T, W2)

    def __str__ (self):
        return "* Multivariate Linear Regression"
        
    def predict(self, X):
        if size(shape(X)) == 1:
            X = reshape(X, (-1,1))
        return dot(X, self.W.T)
