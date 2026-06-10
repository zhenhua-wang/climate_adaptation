import torch
import geopandas as gpd
import torch.nn.functional as F
from src.rbf_gnn_fm import GCDAE, spatial_rff

seed = 1

# load dataset
data_dir = "./data/synthetic"
fine_gdf = gpd.read_file(f"{data_dir}/fine_regions.gpkg")
coarse_gdf = gpd.read_file(f"{data_dir}/coarse_regions.gpkg")
edge_index = torch.load(f"{data_dir}/edge_index.pt")
X_climate = torch.load(f"{data_dir}/X_climate.pt")
X_socio = torch.load(f"{data_dir}/X_socio.pt")
nrow_fine, ncol_fine = 50, 100

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

# train GCDAE: 22 mins 45 secs
config = dict(in_dim=6, cond_dim=c.shape[-1], hidden_dim=512, latent_dim=256,
              modality_idx=modality_idx, dropout=0.2, seed=seed)
model = GCDAE(config)
model.train_model(
    x=X, c=c, edge_index=edge_index,
    n_epochs=2000,
    lr=1e-3
)
model.save_checkpoints("./data/model/")
