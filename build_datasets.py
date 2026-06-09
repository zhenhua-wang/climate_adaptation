import torch
import matplotlib.pyplot as plt
import geopandas as gpd
from py.synthetic_dataset_generation import generate_synthetic_dataset

# generate dataset
nrow_fine, ncol_fine = 50, 100
config = dict(
    nrow_fine=nrow_fine,
    ncol_fine=ncol_fine,
    nrow_coarse=5,
    ncol_coarse=10,
    n_knots=500,
    latent_range=500_000,
    climate_ranges=(800_000, 1_000_000, 1_200_000),
    climate_common_frac=0.3,
    socio_rhos=(0.85, 0.88, 0.9),
    socio_common_frac=0.3,
    group_effect_std=0.3,
    climate_weights1=( 0.5,  0.3,  0.3),
    socio_weights1=(-0.2, -0.1, -0.1),
    climate_weights2=( 0.1,  0.2,  0.1),
    socio_weights2=( 0.3,  0.3,  0.4),
    noise1=0.1,
    noise2=0.1,
)
out_dir = generate_synthetic_dataset(seed=2, config=config, out_dir="./data/synthetic")

# load dataset
data_dir="./data/synthetic"
fine_gdf = gpd.read_file(f"{data_dir}/fine_regions.gpkg")
coarse_gdf = gpd.read_file(f"{data_dir}/coarse_regions.gpkg")
edge_index = torch.load(f"{data_dir}/edge_index.pt")
X_climate = torch.load(f"{data_dir}/X_climate.pt")
X_socio = torch.load(f"{data_dir}/X_socio.pt")

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
