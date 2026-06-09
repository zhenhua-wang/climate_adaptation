from sklearn.metrics import root_mean_squared_error


def print_rmse(name, params, y_test, yhat_test):
    """
    Prints per-target and total RMSE for a model.

    Args:
        name (str): Model name.
        params (str): Parameter description string.
        y_test (ndarray): Shape (N, n_targets), true targets.
        yhat_test (ndarray): Shape (N, n_targets), predicted targets.
    """
    parts = []
    for j in range(y_test.shape[1]):
        rmse = root_mean_squared_error(y_test[:, j], yhat_test[:, j])
        parts.append(f"y{j+1}={rmse:.4f}")
    total = root_mean_squared_error(y_test, yhat_test)
    parts.append(f"total={total:.4f}")
    print(f"{name} ({params}): {'  '.join(parts)}")
