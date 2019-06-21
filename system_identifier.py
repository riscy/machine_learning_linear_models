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
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


def ideal_data(num, dimU, dimY, dimX, noise=1):
    """Linear system data"""
    # generate randomized linear system matrices
    A = np.random.randn(dimX, dimX)
    B = np.random.randn(dimX, dimU)
    C = np.random.randn(dimY, dimX)
    D = np.random.randn(dimY, dimU)

    # make sure state evolution is stable
    U, S, V = np.linalg.svd(A)
    A = np.dot(U, np.dot(np.lib.diag(S / max(S)), V))
    U, S, V = np.linalg.svd(B)
    S2 = np.zeros((np.size(U, 1), np.size(V, 0)))
    S2[:, :np.size(U, 1)] = np.lib.diag(S / max(S))
    B = np.dot(U, np.dot(S2, V))

    # random input
    U = np.random.randn(num, dimU)

    # initial state
    X = np.reshape(np.random.randn(dimX), (1, -1))

    # initial output
    Y = np.reshape(np.dot(C, X[-1]) + np.dot(D, U[0]), (1, -1))

    # generate next state
    X = np.concatenate(
        (X, np.reshape(np.dot(A, X[-1]) + np.dot(B, U[0]), (1, -1))))

    # and so forth
    for u in U[1:]:
        Y = np.concatenate(
            (Y, np.reshape(np.dot(C, X[-1]) + np.dot(D, u), (1, -1))))
        X = np.concatenate(
            (X, np.reshape(np.dot(A, X[-1]) + np.dot(B, u), (1, -1))))

    return U, Y + np.random.randn(num, dimY) * noise


class SystemIdentifier(object):
    """
    Simple Subspace System Identifier
    - U is an n-by-d matrix of "control inputs".
    - Y is an n-by-D matrix of output observations.
    - statedim is the dimension of the internal state variable.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, U, Y, statedim, reg=None):
        if np.size(np.shape(U)) == 1:
            U = np.reshape(U, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0

        yDim = np.size(Y, 1)
        uDim = np.size(U, 1)

        self.output_size = np.size(Y, 1)  # placeholder

        # number of samples of past/future we'll mash together into a 'state'
        width = 1
        # total number of past/future pairings we get as a result
        K = np.size(U, 0) - 2 * width + 1

        # build hankel matrices containing pasts and futures
        U_p = np.array([np.ravel(U[t:t + width]) for t in range(K)]).T
        U_f = np.array([np.ravel(U[t + width:t + 2 * width]) for t in range(K)]).T
        Y_p = np.array([np.ravel(Y[t:t + width]) for t in range(K)]).T
        Y_f = np.array([np.ravel(Y[t + width:t + 2 * width]) for t in range(K)]).T

        # solve the eigenvalue problem
        YfUfT = np.dot(Y_f, U_f.T)
        YfUpT = np.dot(Y_f, U_p.T)
        YfYpT = np.dot(Y_f, Y_p.T)
        UfUpT = np.dot(U_f, U_p.T)
        UfYpT = np.dot(U_f, Y_p.T)
        UpYpT = np.dot(U_p, Y_p.T)
        F = sparse.bmat([
            [None, YfUfT, YfUpT, YfYpT],
            [YfUfT.T, None, UfUpT, UfYpT],
            [YfUpT.T, UfUpT.T, None, UpYpT],
            [YfYpT.T, UfYpT.T, UpYpT.T, None],
        ])
        Ginv = sparse.bmat([
            [np.linalg.pinv(np.dot(Y_f, Y_f.T)), None, None, None],
            [None, np.linalg.pinv(np.dot(U_f, U_f.T)), None, None],
            [None, None, np.linalg.pinv(np.dot(U_p, U_p.T)), None],
            [None, None, None, np.linalg.pinv(np.dot(Y_p, Y_p.T))],
        ])
        F = F - sparse.eye(sp.size(F, 0)) * reg

        # Take smallest eigenvalues
        _, W = sparse_linalg.eigs(Ginv.dot(F), k=statedim, which='SR')

        # State sequence is a weighted combination of the past
        W_U_p = W[width * (yDim + uDim):width * (yDim + uDim + uDim), :]
        W_Y_p = W[width * (yDim + uDim + uDim):, :]
        X_hist = np.dot(W_U_p.T, U_p) + np.dot(W_Y_p.T, Y_p)

        # Regress; trim inputs to match the states we retrieved
        R = np.concatenate((X_hist[:, :-1], U[width:-width].T), 0)
        L = np.concatenate((X_hist[:, 1:], Y[width:-width].T), 0)
        RRi = np.linalg.pinv(np.dot(R, R.T))
        RL = np.dot(R, L.T)
        Sys = np.dot(RRi, RL).T
        self.A = Sys[:statedim, :statedim]
        self.B = Sys[:statedim, statedim:]
        self.C = Sys[statedim:, :statedim]
        self.D = Sys[statedim:, statedim:]

    def __str__(self):
        return "Linear Dynamical System"

    def predict(self, U):
        # If U is a vector, reshape it
        if np.size(np.shape(U)) == 1:
            U = np.reshape(U, (-1, 1))

        # assume some random initial state
        X = np.reshape(np.random.randn(np.size(self.A, 1)), (1, -1))

        # intitial output
        Y = np.reshape(np.dot(self.C, X[-1]) + np.dot(self.D, U[0]), (1, -1))

        # generate next state
        X = np.concatenate((X, np.reshape(np.dot(self.A, X[-1]) + np.dot(self.B, U[0]), (1, -1))))

        # and so forth
        for u in U[1:]:
            Y = np.concatenate((Y, np.reshape(np.dot(self.C, X[-1]) + np.dot(self.D, u), (1, -1))))
            X = np.concatenate((X, np.reshape(np.dot(self.A, X[-1]) + np.dot(self.B, u), (1, -1))))

        return Y
