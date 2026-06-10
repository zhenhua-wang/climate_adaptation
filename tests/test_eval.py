import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import geopandas as gpd
from src.baseline import fit_tune_eval

seed = 1
np.random.seed(seed)

data_dir = "./data/synthetic/"
model_dir = "./data/model/"
fine_gdf = gpd.read_file(f"{data_dir}/fine_regions.gpkg")
embed1, embed2 = torch.load(f"{model_dir}/embed.pt", weights_only=True)
embed = torch.cat([embed1, embed2], dim=-1).numpy()
y = fine_gdf[["y1", "y2", "y3"]].values

all_idx = np.arange(y.shape[0])
group_id = fine_gdf['coarse_id'].values
train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.3, stratify=group_id, random_state=seed)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=2/3, stratify=group_id[temp_idx], random_state=seed)

y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
embed_train = embed[train_idx]
embed_val = embed[val_idx]
embed_test = embed[test_idx]

scaler = StandardScaler().fit(embed_train)
embed_train_norm = scaler.transform(embed_train)
embed_val_norm = scaler.transform(embed_val)
embed_test_norm = scaler.transform(embed_test)


# test evaluation integrity
def test_eval_integrity():
    y_poisoned = y.copy()
    y_poisoned[test_idx] = 99999.0

    yhat1, _ = fit_tune_eval(
        model_fn=lambda p: Ridge(alpha=p["alpha"], solver="lsqr"),
        param_grid=[{"alpha": a} for a in [0.001, 0.01, 0.1, 1, 10, 100, 1000]],
        X_train=embed_train_norm, y_train=y_train,
        X_val=embed_val_norm, y_val=y_val,
        X_test=embed_test_norm, y_test=y_test)

    yhat2, _ = fit_tune_eval(
        model_fn=lambda p: Ridge(alpha=p["alpha"], solver="lsqr"),
        param_grid=[{"alpha": a} for a in [0.001, 0.01, 0.1, 1, 10, 100, 1000]],
        X_train=embed_train_norm, y_train=y_poisoned[train_idx],
        X_val=embed_val_norm, y_val=y_poisoned[val_idx],
        X_test=embed_test_norm, y_test=y_poisoned[test_idx])

    assert np.allclose(yhat1, yhat2), "Ridge: test labels leaked into tuning or training"
