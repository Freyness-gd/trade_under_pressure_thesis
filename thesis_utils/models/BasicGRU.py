from torch import nn


class BasicGRU(nn.Module):
  def __init__(
      self,
      n_features: int,
      hidden_size: int = 128,
      n_layers: int = 2,
      dropout: float = 0.3,
      horizon: int = 1,
  ):
    super(BasicGRU, self).__init__()
    self.hidden_dim = hidden_size
    self.n_layers = n_layers
    self.horizon = horizon

    self.gru = nn.GRU(
      input_size=n_features,
      hidden_size=hidden_size,
      num_layers=n_layers,
      batch_first=True,
      dropout=dropout,
    )
    self.fc = nn.Linear(hidden_size, horizon)

  def forward(self, data):
    out, _ = self.gru(data)
    h_last = out[:, -1, :]

    y_hat = self.fc(h_last)
    return y_hat
