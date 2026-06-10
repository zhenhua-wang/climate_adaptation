from src.synthetic_dataset_generation import generate_synthetic_dataset

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
    climate_weights1=(0.5, 0.3, 0.3),
    socio_weights1=(-0.2, -0.1, -0.1),
    climate_weights2=(0.1, 0.2, 0.1),
    socio_weights2=(0.3, 0.3, 0.4),
    noise1=0.1,
    noise2=0.1,
)
out_dir = generate_synthetic_dataset(seed=2, config=config, out_dir="./data/synthetic")
