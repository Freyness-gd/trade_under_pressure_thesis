from typing import Sequence

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    def __init__(
        self,
        data: DataFrame,
        features: Sequence[str],
        target: str,
        horizon: int = 1,
    ):
        self.X = torch.tensor(data[features].to_numpy(np.float32))

        tgt = data[target].to_numpy(np.float32)
        cols = [np.roll(tgt, -i) for i in range(1, horizon + 1)]
        y_mat = np.stack(cols, axis=1)

        valid = slice(0, len(data) - horizon)
        self.X = self.X[valid]
        self.y = torch.tensor(y_mat[valid])

        self.dyad_idx = torch.tensor(
            data["dyad_idx"].to_numpy(dtype=np.int64)[valid], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx].unsqueeze(0)
        y = self.y[idx]
        di = self.dyad_idx[idx]
        return X, y, di


class DatasetWrapperOptimized(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        features: Sequence[str],
        target: str,
        horizon: int = 1,
    ):
        # Convert to float32 directly from DataFrame to avoid copying unnecessarily
        X_np = data[features].values.astype(np.float32)
        y_np = data[target].values.astype(np.float32)
        dyad_np = data["dyad_idx"].values.astype(np.int64)

        # Compute target windows (shifted columns)
        n_samples = len(data) - horizon
        self.X = torch.from_numpy(X_np[:n_samples])  # shape: [N, D]
        self.y = torch.from_numpy(
            np.column_stack([y_np[i + 1 : i + 1 + n_samples] for i in range(horizon)])
        ).float()  # shape: [N, H]
        self.dyad_idx = torch.from_numpy(dyad_np[:n_samples]).long()  # shape: [N]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return X with shape (1, D) so LSTM sees seq_len=1 (can modify later)
        return self.X[idx].unsqueeze(0), self.y[idx], self.dyad_idx[idx]
