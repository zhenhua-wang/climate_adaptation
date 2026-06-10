import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm


# rff
def spatial_rff(coords_raw, length_scales=[0.1, 0.3, 0.5, 0.8, 1.2], n_rff=64,
                seed=None):
    """
    Spatial random Fourier features of matern 1.5 kernel for list of length_scales.

    Args:
        coords_raw (Tensor): Shape (N, 2), raw spatial coordinates.
        length_scales (list[float]): Length scales for the Matérn kernel.
        n_rff (int): Number of random Fourier features per length scale.
        seed (int, optional): Random seed.

    Returns:
        Tensor: Shape (N, n_rff * len(length_scales)), spatial RFF embeddings.
    """
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
class GCEncoder(nn.Module):
    """
    GCN-based conditional encoder that produces per-modality embeddings.

    Init:
        Args:
            in_dim (int): Dimension of input layer.
            cond_dim (int): Number of conditions.
            hidden_dim (int): Dimension of hidden layer.
            latent_dim (int): Dimension of each modality's latent embedding.
            n_modality (int): Number of modalities.
            dropout (float): Dropout rate for creating corrupted input.
    Forward:
        Args:
            x (tensor): Shape (N, in_dim), node feature matrix.
            c (tensor): Shape (N, cond_dim), condition matrix.
            edge_index (tensor): Shape (2, E), graph structure.
        Return:
            Shape (N, latent_dim), tuple of embedding tensors for each modality.
    """

    def __init__(self, in_dim, cond_dim, hidden_dim, latent_dim, n_modality, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.gcn1 = GCNConv(in_dim + cond_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, latent_dim * n_modality)

    def forward(self, x, c, edge_index):
        xc = torch.cat([F.dropout(x, p=self.dropout, training=self.training), c], dim=-1)
        h = F.relu(self.gcn1(xc, edge_index))
        h = self.gcn2(h, edge_index)
        return self.out(h).split(self.latent_dim, dim=-1)


# decoder
class GCDecoder(nn.Module):
    """
    GCN-based conditional decoder that reconstructs input features from latent embeddings.

    A separate GCN branch is used per modality, each reconstructing its corresponding
    subset of input features indexed by modality_idx.

    Init:
        Args:
            latent_dim (int): Dimension of each modality's latent embedding.
            cond_dim (int): Number of conditions.
            hidden_dim (int): Dimension of hidden layer.
            modality_idx (Tensor): integer tensor of length in_dim assigning each feature to a modality.
            n_modality (int): Number of modalities.

    Forward
        Args:
            zs (tuple[Tensor]): Shape (N, latent_dim), per-modality latent embeddings.
            c (Tensor): Shape (N, cond_dim), condition matrix.
            edge_index (Tensor): Shape (2, E), graph structure.

        Returns:
            Tensor: Shape (N, in_dim), reconstructed feature matrix.
    """

    def __init__(self, latent_dim, cond_dim, hidden_dim, modality_idx, n_modality):
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
class GCDAE(nn.Module):
    """
    Graph Conditional Denoising Autoencoder (GCDAE) for multimodal feature embedding.

    Init:
        Args:
            config (dict): Configuration of GCDAE:
                - in_dim (int): Dimension of input layer.
                - cond_dim (int): Number of conditions.
                - hidden_dim (int): Dimension of hidden layer.
                - latent_dim (int): Dimension of each modality's latent embedding.
                - n_modality (int): Number of modalities.
                - dropout (float): Dropout rate for creating corrupted input.
                - seed (int, optional): Random seed.
    """

    def __init__(self, config: dict):
        torch.manual_seed(config.get("seed", 0))
        super().__init__()
        self.config = config
        self.n_modality = int(config["modality_idx"].max().item()) + 1
        self.encoder = GCEncoder(config["in_dim"], config["cond_dim"],
                                 config["hidden_dim"], config["latent_dim"],
                                 self.n_modality, config.get("dropout", 0.1))
        self.decoder = GCDecoder(config["latent_dim"], config["cond_dim"],
                                 config["hidden_dim"], config["modality_idx"],
                                 self.n_modality)

    def forward(self, x, c, edge_index):
        """
        Args:
            x (tensor): Shape (N, in_dim), node feature matrix.
            c (tensor): Shape (N, cond_dim), condition matrix.
            edge_index (tensor): Shape (2, E), graph structure.
        Return:
            x_hat (tensor): Shape (N, in_dim), reconstructed feature matrix
            zs (tuple[Tensor]): Shape (N, latent_dim), per-modality latent embeddings.
        """
        zs = self.encoder(x, c, edge_index)
        x_hat = self.decoder(zs, c, edge_index)
        return x_hat, zs

    def loss(self, x, x_hat, feat_mask):
        """
        Masked MSE loss of observed features.

        Args:
            x (Tensor): Shape (N, in_dim), true features.
            x_hat (Tensor): Shape (N, in_dim), reconstructed features.
            feat_mask (BoolTensor): Shape (N, in_dim), True where features are non-NaN.

        Returns:
            Tensor: masked MSE loss.
        """
        return F.mse_loss(x_hat[feat_mask], x[feat_mask], reduction="mean")

    def train_model(self, x, c, edge_index, n_epochs=200, lr=1e-3):
        """
        Trains GCDAE using Adam optimizer with cosine annealing LR schedule.
        NaN values in x are zeroed out and excluded from the loss via feat_mask.
        Latent embeddings are stored in self.zs after training.

        Args:
            x (tensor): Shape (N, in_dim), node feature matrix.
            c (tensor): Shape (N, cond_dim), condition matrix.
            edge_index (tensor): Shape (2, E), graph structure.
            n_epochs (int): Number of training epochs.
            lr (float): Initial learning rate.
        """
        feat_mask = ~torch.isnan(x)
        x_input = x.clone()
        x_input[~feat_mask] = 0.0
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

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
        """
        Obtain reconstructed feature and embedding for (new) input.

        Args:
            x (tensor): Shape (N, in_dim), node feature matrix.
            c (tensor): Shape (N, cond_dim), condition matrix.
            edge_index (tensor): Shape (2, E), graph structure.
        Return:
            x_hat (tensor): Shape (N, in_dim), reconstructed feature matrix
            zs (tuple[Tensor]): Shape (N, latent_dim), per-modality latent embeddings.
        """
        feat_mask = ~torch.isnan(x)
        x_input = x.clone()
        x_input[~feat_mask] = 0.0
        self.eval()
        with torch.no_grad():
            x_hat, zs = self.forward(x_input, c, edge_index)
        return x_hat, zs

    def save_checkpoints(self, out_dir):
        """
        Saves model state and latent embeddings to out_dir.

        Args:
            out_dir (str): Path to output directory.
        """
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                    "config": self.config}, f"{out_dir}/model.pt")
        torch.save(self.zs, f"{out_dir}/embed.pt")
