import torch
import numpy as np
import geopandas as gpd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from py.baseline import graph_eigenbasis, IDWRegressor, fit_tune_eval
from py.evaluation import print_rmse

# load datasets
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

data_dir = "./data/synthetic/"
model_dir = "./data/model/"
fine_gdf = gpd.read_file(f"{data_dir}/fine_regions.gpkg")
edge_index = torch.load(f"{data_dir}/edge_index.pt", weights_only=True)
embed1, embed2 = torch.load(f"{model_dir}/embed.pt", weights_only=True)
embed = torch.cat([embed1, embed2], dim=-1).numpy()
y = fine_gdf[["y1", "y2", "y3"]].values
coords = fine_gdf[["x", "y"]].values

# train/val/test splits
all_idx = np.arange(y.shape[0])
group_id = fine_gdf['coarse_id'].values
train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.3, stratify=group_id, random_state=seed)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=2/3, stratify=group_id[temp_idx], random_state=seed)

y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
coord_train, coord_val, coord_test = coords[train_idx], coords[val_idx], coords[test_idx]
embed_train, embed_val, embed_test = embed[train_idx], embed[val_idx], embed[test_idx]

# normalization
scaler = StandardScaler().fit(embed_train)
embed_train_norm = scaler.transform(embed_train)
embed_val_norm = scaler.transform(embed_val)
embed_test_norm = scaler.transform(embed_test)

# model fitting
# idw
yhat_test_idw, best_params = fit_tune_eval(
    model_fn=lambda p: IDWRegressor(power=p["power"]),
    param_grid=[{"power": p} for p in [1, 2, 3, 4, 5]],
    X_train=coord_train, y_train=y_train,
    X_val=coord_val, y_val=y_val,
    X_test=coord_test, y_test=y_test)
print_rmse("idw", f"power={best_params['power']}", y_test, yhat_test_idw)

# knn on coords
yhat_test_knn, best_params = fit_tune_eval(
    model_fn=lambda p: KNeighborsRegressor(n_neighbors=p["k"]),
    param_grid=[{"k": k} for k in [3, 5, 10, 15, 20, 30]],
    X_train=coord_train, y_train=y_train,
    X_val=coord_val, y_val=y_val,
    X_test=coord_test, y_test=y_test)
print_rmse("knn on coords", f"k={best_params['k']}", y_test, yhat_test_knn)

# knn on graph eigenbasis
basis_full = graph_eigenbasis(edge_index, len(fine_gdf), k=200)
yhat_test_knn_basis, best_params = fit_tune_eval(
    model_fn=lambda p: KNeighborsRegressor(n_neighbors=p["k"]),
    param_grid=[{"k": k, "n_basis": n}
                for n in [10, 20, 50, 100, 200, 300]
                for k in [3, 5, 10, 15, 20, 30]],
    X_train=basis_full[train_idx], y_train=y_train,
    X_val=basis_full[val_idx], y_val=y_val,
    X_test=basis_full[test_idx], y_test=y_test)
print_rmse("knn on basis", f"n_basis={best_params['n_basis']}, k={best_params['k']}",
           y_test, yhat_test_knn_basis)

# ridge on embeddings
yhat_test_ridge, best_params = fit_tune_eval(
    model_fn=lambda p: Ridge(alpha=p["alpha"], solver="lsqr"),
    param_grid=[{"alpha": a} for a in [0.001, 0.01, 0.1, 1, 10, 100, 1000]],
    X_train=embed_train_norm, y_train=y_train,
    X_val=embed_val_norm, y_val=y_val,
    X_test=embed_test_norm, y_test=y_test)
print_rmse("ridge with FM", f"alpha={best_params['alpha']}", y_test, yhat_test_ridge)
