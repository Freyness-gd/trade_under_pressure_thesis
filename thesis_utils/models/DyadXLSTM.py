import torch
import torch.nn as nn
from torchxlstm import sLSTM, mLSTM, xLSTM


class DyadXLSTM(nn.Module):
    def __init__(
        self,
        n_features,
        n_dyads,
        num_heads=1,
        embed_dim=32,
        hidden_size=128,
        dropout=0.3,
        horizon=1,
        type="X",
        layers="msm",
    ):
        super(DyadXLSTM, self).__init__()
        self.dyad_embed = nn.Embedding(num_embeddings=n_dyads, embedding_dim=embed_dim)

        if type == "X":
            self.xlstm = xLSTM(
                input_size=n_features + embed_dim,
                head_size=hidden_size,
                num_heads=num_heads,
                batch_first=True,
                layers=layers,
            )
        elif type == "S":
            self.xlstm = sLSTM(
                input_size=n_features + embed_dim,
                head_size=hidden_size,
                num_heads=num_heads,
                batch_first=True,
            )
        elif type == "M":
            self.xlstm = mLSTM(
                input_size=n_features + embed_dim,
                head_size=hidden_size,
                num_heads=num_heads,
                batch_first=True,
            )

        self.fc = nn.Linear(self.xlstm.input_size, horizon)
        self.input_dropout = nn.Dropout(dropout)

    def forward(self, X, dyad_idx):
        embedding = self.dyad_embed(dyad_idx)
        embedding = embedding.unsqueeze(1).expand(-1, X.size(1), -1)
        X = torch.cat([X, embedding], dim=-1)

        X = self.input_dropout(X)

        xlstm_out, _ = self.xlstm(X)
        h_last = xlstm_out[:, -1, :]

        y_hat = self.fc(h_last)
        return y_hat
