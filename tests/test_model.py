import torch
from src.rbf_gnn_fm import GCDAE


# 1. Test one-step training sanity
def test_one_step_no_nan():
    N, in_dim, cond_dim, hidden_dim, latent_dim, n_modality = 20, 6, 4, 16, 8, 2
    modality_idx = torch.tensor([0, 0, 0, 1, 1, 1])
    config = {
        "in_dim": in_dim, "hidden_dim": hidden_dim, "cond_dim": cond_dim,
        "latent_dim": latent_dim, "modality_idx": modality_idx, "seed": 0
    }
    x = torch.randn(N, in_dim)
    c = torch.randn(N, cond_dim)
    edge_index = torch.randint(0, N, (2, 40))
    model = GCDAE(config)
    model.train_model(x=x, c=c, edge_index=edge_index, n_epochs=1)
    x_hat, zs = model(x, c, edge_index)

    assert not torch.isnan(x_hat).any(), "x_hat contains NaN"
    assert not torch.isinf(x_hat).any(), "x_hat contains inf"
    for j, z in enumerate(zs):
        assert not torch.isnan(z).any(), f"zs[{j}] contains NaN"
        assert not torch.isinf(z).any(), f"zs[{j}] contains inf"


# 2. Test embedding exports
def test_embedding_export():
    N, in_dim, cond_dim, hidden_dim, latent_dim, n_modality = 20, 6, 4, 16, 8, 2
    modality_idx = torch.tensor([0, 0, 0, 1, 1, 1])
    config = {
        "in_dim": in_dim, "hidden_dim": hidden_dim, "cond_dim": cond_dim,
        "latent_dim": latent_dim, "modality_idx": modality_idx, "seed": 0
    }
    x = torch.randn(N, in_dim)
    c = torch.randn(N, cond_dim)
    edge_index = torch.randint(0, N, (2, 40))
    model = GCDAE(config)
    model.train_model(x=x, c=c, edge_index=edge_index, n_epochs=1)

    # test embedding shape
    assert len(model.zs) == n_modality
    assert all(z.shape == (N, latent_dim) for z in model.zs)
    # test consistent embedding ordering
    _, zs1 = model.predict(x, c, edge_index)
    _, zs2 = model.predict(x, c, edge_index)
    for z1, z2 in zip(zs1, zs2):
        assert torch.allclose(z1, z2)
