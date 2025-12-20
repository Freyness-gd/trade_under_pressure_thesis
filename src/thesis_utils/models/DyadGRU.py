import torch
from torch import nn


class DyadGRU(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_dyads: int,
        embed_dim: int = 32,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        horizon: int = 1,
    ):
        super(DyadGRU, self).__init__()
        self.hidden_dim = hidden_size
        self.n_layers = n_layers
        self.horizon = horizon

        self.dyad_embed = nn.Embedding(num_embeddings=n_dyads, embedding_dim=embed_dim)

        self.gru = nn.GRU(
            input_size=n_features + embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, X, dyad_idx):
        embedding = self.dyad_embed(dyad_idx)
        embedding = embedding.unsqueeze(1).expand(-1, X.size(1), -1)

        X = torch.cat([X, embedding], dim=-1)

        # run GRU
        gru_out, _ = self.gru(X)
        # take last time step
        h_last = gru_out[:, -1, :]
        # project to horizon
        y_hat = self.fc(h_last)
        return y_hat
