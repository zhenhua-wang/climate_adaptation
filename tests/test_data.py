import os
import hashlib
import torch
import geopandas as gpd
from src.synthetic_dataset_generation import generate_synthetic_dataset


data_dir = "./data/synthetic"
config = torch.load(f"{data_dir}/config.pt", weights_only=False)
out_dir = generate_synthetic_dataset(seed=2, config=config, out_dir="./tests/output")


# 1. Test same synthetic datasets
def file_hash(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def test_pt_files():
    for fname in ["X_climate.pt", "X_socio.pt", "edge_index.pt"]:
        assert file_hash(f"{data_dir}/{fname}") == file_hash(f"{out_dir}/{fname}"), \
            f"Mismatch in {fname}"


def test_gpkg_geometry():
    for fname in ["fine_regions.gpkg", "coarse_regions.gpkg"]:
        gdf1 = gpd.read_file(f"{data_dir}/{fname}")
        gdf2 = gpd.read_file(f"{out_dir}/{fname}")
        assert gdf1.geometry.equals(gdf2.geometry), f"Geometry mismatch in {fname}"


def test_fine_grid_targets():
    fine1 = gpd.read_file(f"{data_dir}/fine_regions.gpkg")
    fine2 = gpd.read_file(f"{out_dir}/fine_regions.gpkg")
    for col in ["y1", "y2", "y3", "x", "y"]:
        assert fine1[col].equals(fine2[col]), f"Mismatch in {col}"


# 2. Schema/shape validity
def test_output_files_exist():
    for fname in ["X_climate.pt", "X_socio.pt", "edge_index.pt",
                  "config.pt", "fine_regions.gpkg", "coarse_regions.gpkg"]:
        assert os.path.exists(f"{out_dir}/{fname}"), f"Missing file: {fname}"


def test_data_shapes():
    X_climate = torch.load(f"{out_dir}/X_climate.pt", weights_only=False)
    X_socio = torch.load(f"{out_dir}/X_socio.pt", weights_only=False)
    edge_index = torch.load(f"{out_dir}/edge_index.pt", weights_only=False)

    assert X_climate.shape == (5000, 3), f"X_climate shape: {X_climate.shape}"
    assert X_socio.shape == (5000, 3), f"X_socio shape: {X_socio.shape}"
    assert edge_index.shape[0] == 2, "edge_index must have 2 rows"
