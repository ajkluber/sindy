import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sympy

from sklearn import linear_model
from sklearn.cross_validation import KFold
from scipy import stats


def iterative_solver_regularized(X, Y, max_iter, centers, f):
    """
        input
            Y: derivative array of size (n, n_dim) [n = number of traj points, n_dim = system dimensionality]
            X: function database array of size (m, n) [m = database dimension]
            lambda: float, sparsity parameter
            max_iter: number of iterations (for LAsso and iterative)
            
        output
            coefficients: array of size (m, n_dim) if regular or Lasso calculation
                                        (max_iter, m, n_dim) if iterative calculation
    """


    # initial guess: least square fitting
    coef = np.linalg.lstsq(X, Y, rcond=1e-10)[0]
    lambda_t = 1.5*np.min(np.abs(coef[coef != 0.0]));
    TRESH = 1e-2;
    N = np.shape(X)[0];

    # initialize lists
    iter_lamb = [lambda_t]
    iter_err = [np.sum((np.dot(X,coef) - Y)**2)]
    iter_coef = [coef]
    iter_force = [np.array([ np.dot(f(i), coef) for i in centers ])]

    for k in range(max_iter):     # loop over number of iterations to consider
        #print 'sparse regression, iteration = ' + str(k)
        coef_old = coef;

        # Enforce sparsity by zeroing coefficients. Resolve regression problem.
        nonzero = np.abs(coef) > 0.0
        if np.sum(nonzero) < 4:
            break

        small_nonzero = (np.abs(coef) <= lambda_t) & nonzero
        big_nonzero = (np.abs(coef) > lambda_t) & nonzero
        coef[small_nonzero] = 0.0
        coef[big_nonzero] = np.linalg.lstsq(X[:,big_nonzero], Y, rcond=1e-10)[0]

        # Stop iteration if there is not further change of coefficients.
        #if (sum(np.abs(coef))<=TRESH):
        #    break

        # Increase sparsity.
        lambda_t = 1.5*np.min(np.abs(coef[coef != 0]));

        error = np.sum((np.dot(X,coef) - Y)**2)
        Force_eff = np.array([ np.dot(f(i), coef) for i in centers ])

        iter_coef.append(np.copy(coef))
        iter_err.append(error)
        iter_lamb.append(lambda_t)
        iter_force.append(Force_eff)

    return np.array(iter_coef), np.array(iter_force), np.array(iter_err), np.array(iter_lamb)


