import hashlib
import torch
import geopandas as gpd
from src.synthetic_dataset_generation import generate_synthetic_dataset


data_dir = "./data/synthetic"
config = torch.load(f"{data_dir}/config.pt", weights_only=False)
out1_dir = generate_synthetic_dataset(seed=2, config=config, out_dir="./tests/output1")
out2_dir = generate_synthetic_dataset(seed=2, config=config, out_dir="./tests/output2")


# Test two synthetic datasets with the same seed by hash
def file_hash(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def test_pt_files():
    for fname in ["X_climate.pt", "X_socio.pt", "edge_index.pt"]:
        assert file_hash(f"{out1_dir}/{fname}") == file_hash(f"{out2_dir}/{fname}"), \
            f"Mismatch in {fname}"


# Test fields in fine-level grids by hash
def var_hash(variable):
    return hashlib.md5(variable.to_numpy().tobytes()).hexdigest()


def test_fine_grid_targets():
    fine1 = gpd.read_file(f"{out1_dir}/fine_regions.gpkg")
    fine2 = gpd.read_file(f"{out2_dir}/fine_regions.gpkg")
    for col in ['coarse_id', 'X_common', 'group_effect',
                'X_climate1', 'X_climate2', 'X_climate3',
                'X_socio1', 'X_socio2', 'X_socio3',
                'y1', 'y2', 'y3']:
        assert var_hash(fine1[col]) == var_hash(fine2[col]), f"Mismatch in {col}"
