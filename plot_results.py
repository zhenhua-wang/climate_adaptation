import numpy as np
import torch
import matplotlib.pyplot as plt
import geopandas as gpd
import torch.nn.functional as F
from src.rbf_gnn_fm import GCDAE, spatial_rff

seed = 1

# load checkpoints
data_dir = "./data/synthetic"
fine_gdf = gpd.read_file(f"{data_dir}/fine_regions.gpkg")
coarse_gdf = gpd.read_file(f"{data_dir}/coarse_regions.gpkg")
edge_index = torch.load(f"{data_dir}/edge_index.pt", weights_only=True)
X_climate = torch.load(f"{data_dir}/X_climate.pt", weights_only=True)
X_socio = torch.load(f"{data_dir}/X_socio.pt", weights_only=True)
nrow_fine, ncol_fine = 50, 100

# plot dataset
x_min, y_min, x_max, y_max = (
    fine_gdf["x"].min(), fine_gdf["y"].min(),
    fine_gdf["x"].max(), fine_gdf["y"].max(),
)
fields = ["X_climate1", "X_climate2", "X_climate3",
          "X_socio1", "X_socio2", "X_socio3",
          "y1", "y2", "y3"]
titles = ["X_climate1 (800km)", "X_climate2 (1000km)", "X_climate3 (1200km)",
          "X_socio1 (rho=0.85)", "X_socio2 (rho=0.88)", "X_socio3 (rho=0.9)",
          "y1 (climate outcome)", "y2 (health outcome)", "y3 (nonlinear health)"]
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle("Synthetic Spatial Dataset", fontsize=14, fontweight="bold")
for ax, field, title in zip(axes.flat, fields, titles):
    values = fine_gdf[field].values.reshape(nrow_fine, ncol_fine)
    im = ax.imshow(values, origin="lower", cmap="plasma",
                   extent=[x_min, x_max, y_min, y_max], aspect="auto")
    coarse_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("./plot/synthetic_dataset.png", dpi=150, bbox_inches="tight")
plt.close()


# prepare inputs
N = X_climate.shape[0]
X = torch.cat([X_climate, X_socio], dim=-1)
modality_idx = torch.tensor([0, 0, 0, 1, 1, 1])

# spatial RFF
coords_raw = torch.tensor(fine_gdf[["x", "y"]].values, dtype=torch.float32)
n_rff = 64
length_scales = [0.1, 0.3, 0.5, 0.8, 1.2]
rff = spatial_rff(coords_raw, length_scales, n_rff, seed=seed)

# one-hot coarse_id condition
coarse_ids = torch.tensor(fine_gdf["coarse_id"].values, dtype=torch.long)
n_coarse = int(coarse_ids.max().item()) + 1
coarse_onehot = F.one_hot(coarse_ids, num_classes=n_coarse).float()
c = torch.cat([rff, coarse_onehot], dim=-1)

model_config = torch.load("./data/model/model.pt", weights_only=True)
model = GCDAE(model_config["config"])
model.load_state_dict(model_config["state_dict"])
model.eval()

# reconstruct
X_hat, _ = model.predict(X, c, edge_index)
X_hat = X_hat.numpy()
X_orig = X.numpy()
print("Reconstruction RMSE: ", np.nanmean((X_orig-X_hat)**2))

# plot reconstruction
feat_titles = [
    "Climate 1 (800km)", "Climate 2 (1000km)", "Climate 3 (1200km)",
    "Socio 1 (ρ=0.85)", "Socio 2 (ρ=0.88)", "Socio 3 (ρ=0.90)",
]
x_min, y_min, x_max, y_max = (
    fine_gdf["x"].min(), fine_gdf["y"].min(),
    fine_gdf["x"].max(), fine_gdf["y"].max(),
)
fig, axes = plt.subplots(6, 3, figsize=(18, 30))
fig.suptitle("GCDAE Reconstruction", fontsize=14, fontweight="bold")
for i, title in enumerate(feat_titles):
    original = X_orig[:, i].reshape(nrow_fine, ncol_fine)
    recon = X_hat[:, i].reshape(nrow_fine, ncol_fine)
    residual = original - recon

    vmin = np.nanmin(original)
    vmax = np.nanmax(original)
    kwargs = dict(origin="lower", extent=[x_min, x_max, y_min, y_max],
                  aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax)

    ax = axes[i, 0]
    im = ax.imshow(original, **kwargs)
    coarse_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_title(f"{title} — Original")
    plt.colorbar(im, ax=ax)

    ax = axes[i, 1]
    im = ax.imshow(recon, **kwargs)
    coarse_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_title(f"{title} — Reconstructed")
    plt.colorbar(im, ax=ax)

    ax = axes[i, 2]
    abs_max = np.nanmax(np.abs(residual))
    im = ax.imshow(residual, origin="lower",
                   extent=[x_min, x_max, y_min, y_max], aspect="auto",
                   cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
    coarse_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_title(f"{title} — Residual")
    plt.colorbar(im, ax=ax)

for ax in axes.flat:
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
plt.tight_layout()
plt.savefig("./plot/reconstruction.png", dpi=150, bbox_inches="tight")
plt.close()
