import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse.linalg
from tqdm import tqdm
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin


def fit_tune_eval(model_fn, param_grid,
                  X_train, y_train,
                  X_val, y_val,
                  X_test, y_test):
    """
    Tune a model over a parameter grid on val set, retrain on train+val, and evaluate on test.

    Args:
        model_fn (callable): A sklearn-compatible model.
        param_grid (list[dict]): List of parameter dicts to search over.
        X_train, y_train: Training features and targets.
        X_val, y_val: Validation features and targets.
        X_test, y_test: Test features and targets.

    Returns:
        yhat_test: Shape (N_test, n_targets), predition results.
        best_params: Parameter dict with lowest RMSE on validation dataset.
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

    Args:
        edge_index (Tensor): Shape (2, E), graph structure.
        num_nodes (int): Number of nodes.
        k (int): Number of eigenvectors.

    Returns:
        Tensor: Shape (N, k), graph eigenbasis.
    """
    lap_index, lap_weight = get_laplacian(
        edge_index, normalization="sym", num_nodes=num_nodes)
    L = to_scipy_sparse_matrix(lap_index, lap_weight, num_nodes=num_nodes)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=k+1, which="SM")
    basis = torch.tensor(eigenvectors[:, 1:], dtype=torch.float32)
    return basis


def distance_matrix(x0, y0, x1, y1):
    """
    Make a distance matrix between pairwise observations.
    Note: from <http://stackoverflow.com/questions/1871536>
    """

    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    # calculate hypotenuse
    return np.hypot(d0, d1)


def simple_idw(x, y, z, xi, yi, power=1):
    """
    Simple inverse distance weighted (IDW) interpolation
    Weights are proportional to the inverse of the distance, so as the distance
    increases, the weights decrease rapidly.
    The rate at which the weights decrease is dependent on the value of power.
    As power increases, the weights for distant points decrease rapidly.
    Note: from <https://gist.github.com/Majramos/5e8985adc467b80cccb0cc22d140634e>
    """

    dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance
    weights = 1.0/(dist+1e-12)**power

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)


class IDWRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for IDW interpolation.
    """

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


class GNNRegressor(nn.Module):
    """
    Multi-layer GCN for spatial regression in a transductive setting.

    Args:
        in_dim (int): Input feature dimensionality.
        hidden_dim (int): Hidden layer size.
        out_dim (int): Number of targets.
        n_layers (int): Number of GCN layers.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden_dim)] +
            [GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers - 1)]
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.out(h)

    def fit(self, x, edge_index, y_train, train_idx, n_epochs=300, lr=1e-3):
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        pbar = tqdm(range(n_epochs), desc="Training GNN")
        for _ in pbar:
            self.train()
            optimizer.zero_grad()
            loss = F.mse_loss(self(x, edge_index)[train_idx], y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return self

    def predict(self, x, edge_index, idx):
        self.eval()
        with torch.no_grad():
            return self(x, edge_index)[idx].numpy()
