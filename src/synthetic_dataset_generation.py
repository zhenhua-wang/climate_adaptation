import os
import torch
import geopandas as gpd
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from shapely.geometry import box
from torch_geometric.utils import grid, get_laplacian, to_scipy_sparse_matrix


def generate_grids(nrow_fine, ncol_fine, nrow_coarse, ncol_coarse):
    """
    Generates two-level spatial grids (fine and coarse) over the contiguous US in EPSG:5070.

    Args:
        nrow_fine (int): Number of rows in the fine grid.
        ncol_fine (int): Number of columns in the fine grid.
        nrow_coarse (int): Number of rows in the coarse grid.
        ncol_coarse (int): Number of columns in the coarse grid.

    Return:
        fine_gdf (GeoDataFrame): Fine-scale grid with centroid coordinates and coarse_id.
        coarse_gdf (GeoDataFrame): Coarse-scale grid.
        edge_index (Tensor): Shape (2, E), graph structure of the fine grid.
        L_fine (spmatrix): Graph Laplacian of the fine grid.
    """
    # create fine scale grids
    x_edges = np.linspace(-2356114.0, 2258154.0, ncol_fine + 1)
    y_edges = np.linspace(268496.0, 3172666.0, nrow_fine + 1)
    cells = [
        box(x_edges[i], y_edges[j], x_edges[i+1], y_edges[j+1])
        for j in range(nrow_fine)
        for i in range(ncol_fine)
    ]
    fine_gdf = gpd.GeoDataFrame(
        {"GEOID": [f"{i:04d}" for i in range(len(cells))]},
        geometry=cells,
        crs="EPSG:5070"
    )
    fine_gdf["x"] = fine_gdf.geometry.centroid.x
    fine_gdf["y"] = fine_gdf.geometry.centroid.y
    # aggregate to coarse scale
    x_coarse = x_edges[::ncol_fine // ncol_coarse]
    y_coarse = y_edges[::nrow_fine // nrow_coarse]
    cells_coarse = [
        box(x_coarse[i], y_coarse[j], x_coarse[i+1], y_coarse[j+1])
        for j in range(nrow_coarse) for i in range(ncol_coarse)
    ]
    coarse_gdf = gpd.GeoDataFrame(geometry=cells_coarse, crs="EPSG:5070")
    # assign coarse id to fine scale grids
    idx = np.arange(nrow_fine * ncol_fine)
    row_c = (idx // ncol_fine) // (nrow_fine // nrow_coarse)
    col_c = (idx % ncol_fine) // (ncol_fine // ncol_coarse)
    fine_gdf["coarse_id"] = row_c * ncol_coarse + col_c
    # build fine graph
    edge_index, _ = grid(height=nrow_fine, width=ncol_fine)
    lap_index, lap_weight = get_laplacian(edge_index, normalization=None)
    L_fine = to_scipy_sparse_matrix(lap_index, lap_weight, num_nodes=nrow_fine * ncol_fine)
    return fine_gdf, coarse_gdf, edge_index, L_fine


def matern_continuous(coords, n_knots, length_scale):
    """
    Samples a spatial random field using a Matérn nu=1.5 kernel approximation.

    Args:
        coords (Tensor): Shape (N, 2), spatial coordinates.
        n_knots (int): Number of knot points sampled from coords.
        length_scale (float): Length scale parameter.

    Return:
        Tensor: Shape (N,), spatial field.
    """
    knots = coords[torch.randperm(len(coords))[:n_knots]]
    w = torch.randn(n_knots)
    diff = coords.unsqueeze(1) - knots.unsqueeze(0)
    dists = torch.norm(diff, dim=2)
    r = (3.0 ** 0.5 * dists) / length_scale
    phi_mat = (1 + r) * torch.exp(-r)
    spatial_field = phi_mat @ w
    return (spatial_field - spatial_field.mean()) / spatial_field.std()


def car_graph(L, rho=0.9):
    """
    Samples a spatial random field from a Conditional Autoregressive (CAR) model.

    Args:
        L (spmatrix): Graph Laplacian of the fine grid.
        rho (float): Spatial autocorrelation strength.

    Return:
        Tensor: Shape (N,), CAR random field.
    """
    n = L.shape[0]
    A = scipy.sparse.diags(L.diagonal()) - L
    D_inv = scipy.sparse.diags(1.0 / L.diagonal())
    W = D_inv @ A
    Q = scipy.sparse.eye(n) - rho * W
    z = np.random.randn(n)
    x, _ = scipy.sparse.linalg.cg(Q, z, atol=1e-6)
    x = torch.tensor(x, dtype=torch.float32)
    return (x - x.mean()) / x.std()


def make_mask(n, coarse_id, n_coarse, n_missing=5):
    """
    Creates a structured missing data mask over three feature columns,
    where the first column is fully observed, and the second and third columns
    each have n_missing randomly selected coarse regions masked out.

    Args:
        n (int): Number of fine-scale nodes.
        coarse_id (ndarray): Shape (N,), coarse region index for each fine node.
        n_coarse (int): Total number of coarse regions.
        n_missing (int): Number of coarse regions to mask per column.

    Returns:
        ndarray: Shape (N, 3), true where features are observed.
    """
    missing1 = np.random.choice(n_coarse, n_missing, replace=False)
    missing2 = np.random.choice(n_coarse, n_missing, replace=False)
    return np.stack([
        np.ones(n, dtype=bool),
        ~np.isin(coarse_id, missing1),
        ~np.isin(coarse_id, missing2),
    ], axis=1)


def generate_synthetic_dataset(seed: int, config: dict, out_dir: str) -> str:
    """
    Constructs fine and coarse grids, simulates climate (Matérn) and socioeconomic
    (CAR) spatial fields with a shared latent component, and produces three regression
    targets with varying linear and nonlinear relationships. Structured missing patterns
    are applied to both modalities. All outputs are saved to out_dir.

    Args:
        seed (int): Random seed for reproducibility.
        config (dict): Configuration dictionary with keys:
            - nrow_fine, ncol_fine (int): Fine grid dimensions.
            - nrow_coarse, ncol_coarse (int): Coarse grid dimensions.
            - n_knots (int): Number of knots for Matérn fields.
            - latent_range (float, optional): Length scale for shared latent field.
            - climate_ranges (tuple[float], optional): Length scales for climate fields.
            - climate_common_frac (float, optional): Fraction of shared latent in climate fields.
            - socio_rhos (tuple[float], optional): CAR autocorrelation values for socio fields.
            - socio_common_frac (float, optional): Fraction of shared latent in socio fields.
            - group_effect_std (float, optional): Std of coarse-level group effect.
            - climate_weights1/2 (tuple[float], optional): Climate feature weights for y1/y2.
            - socio_weights1/2 (tuple[float], optional): Socio feature weights for y1/y2.
            - noise1, noise2 (float, optional): Noise std for targets.
        out_dir (str): Path to output directory.

    Save:
        - fine_regions.gpkg: Fine grid with features and targets.
        - coarse_regions.gpkg: Coarse grid.
        - edge_index.pt: Fine grid graph structure.
        - X_climate.pt, X_socio.pt: Feature tensors with missing values.
        - config.pt: Config dictionary.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # unpack config
    nrow_fine = config["nrow_fine"]
    ncol_fine = config["ncol_fine"]
    nrow_coarse = config["nrow_coarse"]
    ncol_coarse = config["ncol_coarse"]
    n_knots = config["n_knots"]
    latent_range = config.get("latent_range", 500_000)
    climate_ranges = config.get("climate_ranges", (800_000, 1_000_000, 1_200_000))
    climate_common_frac = config.get("climate_common_frac", 0.3)
    socio_rhos = config.get("socio_rhos", (0.7, 0.8, 0.9))
    socio_common_frac = config.get("socio_common_frac", 0.3)
    group_effect_std = config.get("group_effect_std", 0.3)
    climate_weights1 = config.get("climate_weights1", (0.5, 0.3, 0.2))
    climate_weights2 = config.get("climate_weights2", (0.1, 0.3, 0.6))
    socio_weights1 = config.get("socio_weights1", (0.5, 0.3, 0.2))
    socio_weights2 = config.get("socio_weights2", (0.2, 0.3, 0.5))
    noise1 = config.get("noise1", 0.2)
    noise2 = config.get("noise2", 0.2)

    # generate grids
    fine_gdf, coarse_gdf, edge_index, L_fine = generate_grids(
        nrow_fine, ncol_fine, nrow_coarse, ncol_coarse)
    coords = torch.tensor(fine_gdf[["x", "y"]].values, dtype=torch.float32)
    n = len(coords)
    n_coarse = len(coarse_gdf)
    coarse_id = fine_gdf["coarse_id"].values

    # shared latent field
    latent_cont = matern_continuous(coords, n_knots, length_scale=latent_range)

    # climate modality
    X_climate = torch.zeros(n, len(climate_ranges))
    for i, r in enumerate(climate_ranges):
        f = matern_continuous(coords, n_knots, length_scale=r)
        X_climate[:, i] = (1 - climate_common_frac) * f + climate_common_frac * latent_cont

    # socio modality: 3 CAR random fields at finer-level + group effect
    coarse_id_tensor = torch.tensor(coarse_id, dtype=torch.long)
    group_effect = group_effect_std * torch.randn(n_coarse)[coarse_id_tensor]
    X_socio = torch.zeros(n, len(socio_rhos))
    for i, rho in enumerate(socio_rhos):
        f_fine = car_graph(L_fine, rho=rho)
        X_socio[:, i] = (1 - socio_common_frac) * f_fine + socio_common_frac * latent_cont +\
            group_effect

    # target
    y1 = X_climate@torch.tensor(climate_weights1) + X_socio@torch.tensor(socio_weights1) + noise1*torch.randn(n)
    y2 = X_climate@torch.tensor(climate_weights2) + X_socio@torch.tensor(socio_weights2) + noise2*torch.randn(n)
    y3 = torch.exp(X_climate@torch.tensor(climate_weights2)) + 0.2*(X_socio@torch.tensor(socio_weights2))**2 + noise2*torch.randn(n)

    # create structured missing pattern on features only
    mask_climate = make_mask(n, coarse_id, n_coarse)
    mask_socio = make_mask(n, coarse_id, n_coarse)
    for j in range(3):
        X_climate[~mask_climate[:, j], j] = float("nan")
        X_socio[~mask_socio[:, j], j] = float("nan")

    # save datasets
    fine_gdf["X_common"] = latent_cont.numpy()
    fine_gdf["group_effect"] = group_effect.numpy()
    for j in range(3):
        fine_gdf[f"X_climate{j+1}"] = X_climate[:, j].numpy()
        fine_gdf[f"X_socio{j+1}"] = X_socio[:, j].numpy()
    fine_gdf["y1"] = y1.numpy()
    fine_gdf["y2"] = y2.numpy()
    fine_gdf["y3"] = y3.numpy()

    fine_gdf.to_file(os.path.join(out_dir, "fine_regions.gpkg"), driver="GPKG")
    coarse_gdf.to_file(os.path.join(out_dir, "coarse_regions.gpkg"), driver="GPKG")
    torch.save(edge_index, os.path.join(out_dir, "edge_index.pt"))
    torch.save(X_climate, os.path.join(out_dir, "X_climate.pt"))
    torch.save(X_socio, os.path.join(out_dir, "X_socio.pt"))
    torch.save(config, os.path.join(out_dir, "config.pt"))

    return out_dir
