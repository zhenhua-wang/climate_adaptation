import torch
import numpy as np
import geopandas as gpd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

# load checkpoints
data_dir = "./data/synthetic/"
model_dir = "./data/model/"
fine_gdf = gpd.read_file(f"{data_dir}/fine_regions.gpkg")
edge_index = torch.load(f"{data_dir}/edge_index.pt", weights_only=True)
embed1, embed2 = torch.load(f"{model_dir}/embed.pt", weights_only=True)
embed = torch.cat([embed1, embed2], dim=-1).numpy()
y = fine_gdf[["y1", "y2", "y3"]].values
coords = fine_gdf[["x", "y"]].values

# split datasets
all_idx = np.arange(y.shape[0])
group_id = fine_gdf['coarse_id'].values
train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.3, stratify=group_id, random_state=seed)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=2/3, stratify=group_id[temp_idx], random_state=seed)

y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
embed_train, embed_val, embed_test = embed[train_idx], embed[val_idx], embed[test_idx]
coord_train, coord_val, coord_test = coords[train_idx], coords[val_idx], coords[test_idx]

# normalization
scaler = StandardScaler().fit(embed_train)
embed_train_norm = scaler.transform(embed_train)
embed_val_norm = scaler.transform(embed_val)
embed_test_norm = scaler.transform(embed_test)

embed_trainval_norm = np.concatenate([embed_train_norm, embed_val_norm])
coord_trainval = np.concatenate([coord_train, coord_val])
y_trainval = np.concatenate([y_train, y_val])

# tune knn on val
best_k, best_rmse = None, float("inf")
for k in [3, 5, 10, 15, 20, 30]:
    knn = KNeighborsRegressor(n_neighbors=k).fit(coord_train, y_train)
    rmse = root_mean_squared_error(y_val, knn.predict(coord_val))
    if rmse < best_rmse:
        best_rmse, best_k = rmse, k

knn = KNeighborsRegressor(n_neighbors=best_k).fit(coord_trainval, y_trainval)
yhat_test_knn = knn.predict(coord_test)
rmse_knn = root_mean_squared_error(y_test, yhat_test_knn)
print(f"knn  (k={best_k}): rmse={rmse_knn:.4f}")

# tune ridge on val
best_alpha, best_rmse = None, float("inf")
for alpha in np.logspace(-3, 3, 10):
    ridge = Ridge(alpha=alpha, solver="lsqr").fit(embed_train_norm, y_train)
    rmse = root_mean_squared_error(y_val, ridge.predict(embed_val_norm))
    if rmse < best_rmse:
        best_rmse, best_alpha = rmse, alpha

ridge = Ridge(alpha=best_alpha, solver="lsqr").fit(embed_trainval_norm, y_trainval)
yhat_test_ridge = ridge.predict(embed_test_norm)
rmse_ridge = root_mean_squared_error(y_test, yhat_test_ridge)
print(f"ridge (alpha={best_alpha:.4f}): rmse={rmse_ridge:.4f}")
