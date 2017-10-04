"""Machine learning with linear models - a demo
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

My Ph.D. supervisor Dr. Michael Bowling introduced me to RRR; Dr. Tijl De Bie
gave a great talk on subspace system identification in 2005 that I modeled my
implementation on: http://videolectures.net/slsfs05_bie_slasi/
"""

import multivariate_regressor
import reduced_rank_regressor
import system_identifier

from scipy import around
from scipy import size
from scipy.linalg import norm
from numpy.random import seed

DIM_X = 20                      # dimensionality of input
DIM_Y = 15                      # dimensionality of output
RANK = 10                       # internal rank/bottleneck
NOISE_SCALE = 1.0
NUM_SAMPLES = 5000
SPLIT = int(NUM_SAMPLES/2)      # train/test split
REG = 1e-6                      # regularization on the model


def sqerr(matrix1, matrix2):
    """Squared error (frobenius norm of diff) between two matrices."""
    return around(pow(norm(matrix1 - matrix2, 'fro'), 2) / size(matrix2, 0), 5)


if __name__ == '__main__':
    seed(10)
    for model in [multivariate_regressor, reduced_rank_regressor, system_identifier]:
        # generate the data:
        print(model.ideal_data.__doc__)
        XX, YY = model.ideal_data(NUM_SAMPLES, DIM_X, DIM_Y, RANK, NOISE_SCALE)
        # run each of the regressors against it:
        for regressor in ([
                multivariate_regressor.MultivariateRegressor(XX[:SPLIT], YY[:SPLIT], REG),
                reduced_rank_regressor.ReducedRankRegressor(XX[:SPLIT], YY[:SPLIT], RANK, REG),
                system_identifier.SystemIdentifier(XX[:SPLIT], YY[:SPLIT], RANK, REG)]):
            print('  {}'.format(regressor))
            training_error = sqerr(regressor.predict(XX[:SPLIT]), YY[:SPLIT])
            testing_error = sqerr(regressor.predict(XX[SPLIT+1:]), YY[SPLIT+1:])
            print('    Training error: {}\n    Testing error: {}'
                  .format(training_error, testing_error))
