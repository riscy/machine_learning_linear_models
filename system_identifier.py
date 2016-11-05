"""
Subspace system identification with regularization.
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Based on a talk on Subspace System Identification by Tijl De Bie (2005):

Assume every output (y_i) is a function of the input (u_i) and
the current state x_i of the system, i.e.,
   y_i = dot(C, x_i) + dot(D, u_i)
Also assume the system state evolves after every input:
   x_(i+1) = dot(A, x_i) + dot(B, u_i)
This is a linear dynamical system.
"""

from numpy.lib import diag
from scipy import randn
from scipy import dot
from scipy import shape
from scipy import reshape
from scipy import size
from scipy import zeros
from scipy import array
from scipy import bmat
from scipy import argsort
from scipy import ravel
from scipy import concatenate
from scipy.sparse import eye
from scipy.linalg import pinv
from scipy.linalg import eig
from scipy.linalg import svd

def ideal_data(num, dimU, dimY, dimX, noise=1):
    """Linear system data"""
    # generate randomized linear system matrices
    A = randn(dimX, dimX)
    B = randn(dimX, dimU)
    C = randn(dimY, dimX)
    D = randn(dimY, dimU)

    # make sure state evolution is stable
    U, S, V = svd(A)
    A = dot(U, dot(diag(S / max(S)), V))
    U, S, V = svd(B)
    S2 = zeros((size(U,1), size(V,0)))
    S2[:,:size(U,1)] = diag(S / max(S))
    B = dot(U, dot(S2, V))

    # random input
    U = randn(num, dimU)

    # initial state
    X = reshape(randn(dimX), (1,-1))

    # initial output
    Y = reshape(dot(C, X[-1]) + dot(D, U[0]), (1,-1))

    # generate next state
    X = concatenate((X, reshape(dot(A, X[-1]) + dot(B, U[0]), (1,-1))))

    # and so forth
    for u in U[1:]:
        Y = concatenate((Y, reshape(dot(C, X[-1]) + dot(D, u), (1,-1))))
        X = concatenate((X, reshape(dot(A, X[-1]) + dot(B, u), (1,-1))))

    return U, Y + randn(num, dimY) * noise


def create_model(U, Y, dimensionality, regularization):
    return SystemIdentifier(U, Y, dimensionality, regularization)


class SystemIdentifier(object):
    """
    Simple Subspace System Identifier
    - U is an n-by-d matrix of "control inputs".
    - Y is an n-by-D matrix of output observations.
    - statedim is the dimension of the internal state variable.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, U, Y, statedim, reg=None):
        if size(shape(U)) == 1:
            U = reshape(U, (-1,1))
        if size(shape(Y)) == 1:
            Y = reshape(Y, (-1,1))
        if reg is None:
            reg = 0

        yDim = size(Y,1)
        uDim = size(U,1)

        self.output_size = size(Y,1) # placeholder

        # number of samples of past/future we'll mash together into a 'state'
        width = 1
        # total number of past/future pairings we get as a result
        K = size(U,0) - 2 * width + 1

        # build hankel matrices containing pasts and futures
        U_p = array([ravel(U[t : t + width]) for t in range(K)]).T
        U_f = array([ravel(U[t + width : t + 2 * width]) for t in range(K)]).T
        Y_p = array([ravel(Y[t : t + width]) for t in range(K)]).T
        Y_f = array([ravel(Y[t + width : t + 2 * width]) for t in range(K)]).T

        # solve the eigenvalue problem
        YfUfT = dot(Y_f, U_f.T)
        YfUpT = dot(Y_f, U_p.T)
        YfYpT = dot(Y_f, Y_p.T)
        UfUpT = dot(U_f, U_p.T)
        UfYpT = dot(U_f, Y_p.T)
        UpYpT = dot(U_p, Y_p.T)
        F = array(bmat([[zeros((yDim*width,yDim*width)), YfUfT, YfUpT, YfYpT],
                        [YfUfT.T, zeros((uDim*width,uDim*width)), UfUpT, UfYpT],
                        [YfUpT.T, UfUpT.T, zeros((uDim*width,uDim*width)), UpYpT ],
                        [YfYpT.T, UfYpT.T, UpYpT.T, zeros((yDim*width,yDim*width))]]))
        G = array(bmat([[dot(Y_f,Y_f.T), zeros((yDim*width,uDim*width)),
                         zeros((yDim*width,uDim*width)), zeros((yDim*width,yDim*width))],
                        [zeros((uDim*width,yDim*width)), dot(U_f,U_f.T),
                         zeros((uDim*width,uDim*width)), zeros((uDim*width,yDim*width))],
                        [zeros((uDim*width,yDim*width)), zeros((uDim*width,uDim*width)),
                         dot(U_p,U_p.T), zeros((uDim*width,yDim*width))],
                        [zeros((yDim*width,yDim*width)), zeros((yDim*width,uDim*width)),
                         zeros((yDim*width,uDim*width)), dot(Y_p,Y_p.T)]]))
        F = F - eye(size(F, 0)) * reg

        # Take smallest eigenvalues
        V,W = eig(dot(pinv(G), F))
        W = W[:, argsort(V)[:statedim]]

        # State sequence is a weighted combination of the past
        W_U_p = W[ width * (yDim + uDim) : width * (yDim + uDim + uDim), :]
        W_Y_p = W[ width * (yDim + uDim + uDim):, :]
        X_hist = dot(W_U_p.T, U_p) + dot(W_Y_p.T, Y_p)

        # Regress; trim inputs to match the states we retrieved
        R = concatenate((X_hist[:, :-1], U[width:-width].T), 0)
        L = concatenate((X_hist[:, 1: ], Y[width:-width].T), 0)
        RRi = pinv(dot(R, R.T))
        RL  = dot(R, L.T)
        Sys = dot(RRi, RL).T
        self.A = Sys[:statedim, :statedim]
        self.B = Sys[:statedim, statedim:]
        self.C = Sys[statedim:, :statedim]
        self.D = Sys[statedim:, statedim:]


    def __str__ (self):
        return "Linear Dynamical System"


    def predict(self, U):
        # If U is a vector, reshape it
        if size(shape(U)) == 1:
            U = reshape(U, (-1, 1))

        # assume some random initial state
        X = reshape(randn(size(self.A, 1)), (1, -1))

        # intitial output
        Y = reshape(dot(self.C, X[-1]) + dot(self.D, U[0]), (1, -1))

        # generate next state
        X = concatenate((X, reshape(dot(self.A, X[-1]) + dot(self.B, U[0]), (1, -1))))

        # and so forth
        for u in U[1:]:
            Y = concatenate((Y, reshape(dot(self.C, X[-1]) + dot(self.D, u), (1, -1))))
            X = concatenate((X, reshape(dot(self.A, X[-1]) + dot(self.B, u), (1, -1))))

        return Y
