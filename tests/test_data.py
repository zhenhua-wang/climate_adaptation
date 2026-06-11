import os
import hashlib
import torch
import geopandas as gpd
from src.synthetic_dataset_generation import generate_synthetic_dataset


data_dir = "./data/synthetic"
config = torch.load(f"{data_dir}/config.pt", weights_only=False)
out_dir = generate_synthetic_dataset(seed=2, config=config, out_dir="./tests/output")


# 1. Test same synthetic datasets
def test_pt_files():
    for fname in ["X_climate.pt", "X_socio.pt", "edge_index.pt"]:
        saved = torch.load(f"{data_dir}/{fname}", weights_only=False)
        generated = torch.load(f"{out_dir}/{fname}", weights_only=False)
        assert torch.allclose(
            saved, generated,
            # tolerance are set according to https://stackoverflow.com/questions/75622268/comparing-two-tensors-in-pytorch
            atol=1e-4, rtol=1e-4,
            equal_nan=True), f"Values are not close in {fname}"


def test_gpkg_geometry():
    for fname in ["fine_regions.gpkg", "coarse_regions.gpkg"]:
        gdf1 = gpd.read_file(f"{data_dir}/{fname}")
        gdf2 = gpd.read_file(f"{out_dir}/{fname}")
        assert gdf1.geometry.equals(gdf2.geometry), f"Geometry mismatch in {fname}"


def test_fine_grid_fields():
    fine1 = gpd.read_file(f"{data_dir}/fine_regions.gpkg")
    fine2 = gpd.read_file(f"{out_dir}/fine_regions.gpkg")
    for col in ['coarse_id', 'X_common', 'group_effect',
                'X_climate1', 'X_climate2', 'X_climate3',
                'X_socio1', 'X_socio2', 'X_socio3',
                'y1', 'y2', 'y3']:
        assert torch.allclose(
            torch.as_tensor(fine1[col].to_numpy()),
            torch.as_tensor(fine2[col].to_numpy()),
            # tolerance are set according to https://stackoverflow.com/questions/75622268/comparing-two-tensors-in-pytorch
            atol=1e-4, rtol=1e-4,
            equal_nan=True), f"Values are not close in {col}"


# 2. Schema/shape validity
def test_output_files_exist():
    for fname in ["X_climate.pt", "X_socio.pt", "edge_index.pt",
                  "config.pt", "fine_regions.gpkg", "coarse_regions.gpkg"]:
        assert os.path.exists(f"{out_dir}/{fname}"), f"Missing file: {fname}"


def test_data_shapes():
    X_climate = torch.load(f"{out_dir}/X_climate.pt", weights_only=False)
    X_socio = torch.load(f"{out_dir}/X_socio.pt", weights_only=False)
    edge_index = torch.load(f"{out_dir}/edge_index.pt", weights_only=False)
    fine = gpd.read_file(f"{out_dir}/fine_regions.gpkg")
    coarse = gpd.read_file(f"{out_dir}/coarse_regions.gpkg")

    assert X_climate.shape == (5000, 3), f"X_climate shape: {X_climate.shape}"
    assert X_socio.shape == (5000, 3), f"X_socio shape: {X_socio.shape}"
    assert edge_index.shape[0] == 2, "edge_index must have 2 rows"
    assert fine.shape[0] == 5000, "fine-level data must have 5000 grids."
    assert coarse.shape[0] == 50, "coarse-level data must have 50 grids."
