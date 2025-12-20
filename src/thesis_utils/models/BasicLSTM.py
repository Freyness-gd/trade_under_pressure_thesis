from torch import nn


class BasicLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        horizon: int = 1,
    ):
        super(BasicLSTM, self).__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, data):
        lstm_out, _ = self.lstm(data)
        h_last = lstm_out[:, -1, :]
        y_hat = self.fc(h_last)
        return y_hat
