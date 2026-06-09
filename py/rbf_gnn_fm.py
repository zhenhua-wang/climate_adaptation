import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph
from tqdm import tqdm

# rff
def spatial_rff(coords_raw, length_scales=[0.1, 0.3, 0.5, 0.8, 1.2], n_rff=64,
                seed=None):
    """ Spatial random Fourier features for matern 1.5 kernel"""
    if seed is not None:
        torch.manual_seed(seed)
    coords = (coords_raw - coords_raw.mean(0)) / coords_raw.std(0)
    feats = []
    for ls in length_scales:
        W = torch.randn(n_rff, 2) * (3**0.5) / ls
        b = torch.rand(n_rff) * 2 * torch.pi
        feats.append((2/n_rff)**0.5 * torch.cos(coords @ W.T + b))
    return torch.cat(feats, dim=-1)

# encoder
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, cond_dim, latent_dim, n_modality, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.gcn1 = GCNConv(in_dim + cond_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, latent_dim * n_modality)

    def forward(self, x, c, edge_index):
        xc = torch.cat([F.dropout(x, p=self.dropout, training=self.training), c], dim=-1)
        h = F.relu(self.gcn1(xc, edge_index))
        h = self.gcn2(h, edge_index)
        return self.proj(h).split(self.latent_dim, dim=-1)


# decoder
class GCNDecoder(nn.Module):
    def __init__(self, hidden_dim, cond_dim, latent_dim, modality_idx, n_modality):
        super().__init__()
        self.modality_idx = modality_idx
        self.n_modality = n_modality
        self.gcn1s = nn.ModuleList()
        self.gcn2s = nn.ModuleList()
        self.outs = nn.ModuleList()
        for k in range(n_modality):
            self.gcn1s.append(GCNConv(latent_dim + cond_dim, hidden_dim))
            self.gcn2s.append(GCNConv(hidden_dim, hidden_dim))
            self.outs.append(nn.Linear(hidden_dim, int((modality_idx == k).sum().item())))

    def forward(self, zs, c, edge_index):
        N = zs[0].shape[0]
        x_hat = torch.empty(N, self.modality_idx.shape[0],
                            device=zs[0].device, dtype=zs[0].dtype)
        for k in range(self.n_modality):
            h = F.relu(self.gcn1s[k](torch.cat([zs[k], c], dim=-1), edge_index))
            h = F.relu(self.gcn2s[k](h, edge_index))
            x_hat[:, self.modality_idx == k] = self.outs[k](h)
        return x_hat


# DAE
class GCNDAE(nn.Module):
    def __init__(self, config: dict):
        torch.manual_seed(config.get("seed", 0))
        super().__init__()
        self.config = config
        self.n_modality = int(config["modality_idx"].max().item()) + 1
        self.encoder = GCNEncoder(config["in_dim"], config["hidden_dim"],
                                  config["cond_dim"], config["latent_dim"],
                                  self.n_modality, config.get("dropout", 0.1))
        self.decoder = GCNDecoder(config["hidden_dim"], config["cond_dim"],
                                  config["latent_dim"], config["modality_idx"],
                                  self.n_modality)

    def forward(self, x, c, edge_index):
        zs = self.encoder(x, c, edge_index)
        x_hat = self.decoder(zs, c, edge_index)
        return x_hat, zs

    def loss(self, x, x_hat, feat_mask):
        return F.mse_loss(x_hat[feat_mask], x[feat_mask], reduction="mean")

    def train_model(self, data, n_epochs=200, lr=1e-3):
        x, c, edge_index = data["x"], data["c"], data["edge_index"]
        feat_mask = ~torch.isnan(x)
        x_input = x.clone()
        x_input[~feat_mask] = 0.0
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs)

        self.train()
        pbar = tqdm(range(n_epochs), desc="Training")
        for _ in pbar:
            optimizer.zero_grad()
            x_hat, _ = self.forward(x_input, c, edge_index)
            recon = self.loss(x_input, x_hat, feat_mask)
            recon.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{recon.item():.4f}")
        _, self.zs = self.predict(x, c, edge_index)

    def predict(self, x, c, edge_index):
        feat_mask = ~torch.isnan(x)
        x_input = x.clone()
        x_input[~feat_mask] = 0.0
        self.eval()
        with torch.no_grad():
            x_hat, zs = self.forward(x_input, c, edge_index)
        return x_hat, zs


    def save_checkpoints(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        # model states
        torch.save({"state_dict": self.state_dict(),
                    "config": self.config}, f"{out_dir}/model.pt")
        # embeddings
        torch.save(self.zs, f"{out_dir}/embed.pt")
