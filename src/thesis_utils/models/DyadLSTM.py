import torch
import torch.nn as nn


class DyadLSTM(nn.Module):
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
        super(DyadLSTM, self).__init__()
        self.horizon = horizon

        # 1) dyad embedding layer
        self.dyad_embed = nn.Embedding(num_embeddings=n_dyads, embedding_dim=embed_dim)

        # 2) LSTM over [features + embedding]
        self.lstm = nn.LSTM(
            input_size=n_features + embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )

        # 3) final projection
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, X, dyad_idx):
        embedding = self.dyad_embed(dyad_idx)
        embedding = embedding.unsqueeze(1).expand(-1, X.size(1), -1)

        X = torch.cat([X, embedding], dim=-1)

        # run LSTM
        lstm_out, _ = self.lstm(X)
        # take last time step
        h_last = lstm_out[:, -1, :]
        # project to horizon
        y_hat = self.fc(h_last)
        return y_hat
