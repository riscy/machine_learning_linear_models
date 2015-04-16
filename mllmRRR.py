#
# Reduced rank regression
# Requires pylab to be installed.
#
# Implemented by Chris Rayner (2015)
# dchrisrayner AT gmail DOT com
#
# Optimal linear 'bottlenecking' or 'multitask learning'.
#
# URL: http://www.cs.ualberta.ca/~rayner/
# Git: https://github.com/riscy/mllm

from pylab import *

def idealData(num, dimX, dimY, rrank, noise=1):
    """
    Example of ideal data for this model.
    """

    X = randn(num, dimX)
    W = dot(randn(dimX, rrank), randn(rrank, dimY))
    Y = dot(X, W) + randn(num, dimY) * noise

    return X, Y;


class mllmRRR:
    """
    Reduced Rank Regression (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """

    def __init__ (self, X, Y, rrank, reg=None):

        if size(shape(X)) == 1:
            X = reshape(X, (-1,1))
        if size(shape(Y)) == 1:
            Y = reshape(Y, (-1,1))
        if reg is None:
           reg = 0

        CXX = dot(X.T, X) + reg * eye(size(X, 1))
        CXY = dot(X.T, Y)
        U,S,V = svd(dot(CXY.T, dot(pinv(CXX), CXY)))
        self.W = V[0:rrank, :].T
        self.A = dot(pinv(CXX), dot(CXY, self.W)).T


    def __str__ (self):
        return "* Reduced Rank Regression"

        
    def predict(self, X):
        if size(shape(X)) == 1:
            X = reshape(X, (-1,1))
        return dot(X, dot(self.A.T, self.W.T))
