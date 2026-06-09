import torch
import numpy as np
import scipy.sparse.linalg
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin


def fit_tune_eval(model_fn, param_grid,
                  X_train, y_train,
                  X_val, y_val,
                  X_test, y_test):
    """
    Tune a model over a parameter grid on val set, retrain on train+val, and evaluate on test.
    """
    best_params, best_rmse = None, float("inf")
    for params in param_grid:
        model = model_fn(params).fit(X_train, y_train)
        rmse = root_mean_squared_error(y_val, model.predict(X_val))
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params

    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    model = model_fn(best_params).fit(X_trainval, y_trainval)
    yhat_test = model.predict(X_test)
    return yhat_test, best_params


def graph_eigenbasis(edge_index, num_nodes, k=50):
    """
    Computes the k smallest eigenvectors of the graph Laplacian.
    """
    lap_index, lap_weight = get_laplacian(
        edge_index, normalization="sym", num_nodes=num_nodes)
    L = to_scipy_sparse_matrix(lap_index, lap_weight, num_nodes=num_nodes)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=k+1, which="SM")
    basis = torch.tensor(eigenvectors[:, 1:], dtype=torch.float32)
    return basis


def distance_matrix(x0, y0, x1, y1):
    """ Make a distance matrix between pairwise observations.
    Note: from <http://stackoverflow.com/questions/1871536>
    """

    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    # calculate hypotenuse
    return np.hypot(d0, d1)


def simple_idw(x, y, z, xi, yi, power=1):
    """ Simple inverse distance weighted (IDW) interpolation
    Weights are proportional to the inverse of the distance, so as the distance
    increases, the weights decrease rapidly.
    The rate at which the weights decrease is dependent on the value of power.
    As power increases, the weights for distant points decrease rapidly.
    """

    dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance
    weights = 1.0/(dist+1e-12)**power

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)


class IDWRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, power=2):
        self.power = power

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        return simple_idw(
            self.X_train_[:, 0], self.X_train_[:, 1], self.y_train_,
            X[:, 0], X[:, 1], power=self.power)
