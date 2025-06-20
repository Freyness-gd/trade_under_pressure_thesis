from typing import Sequence

import numpy as np
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

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx].unsqueeze(0)
        y = self.y[idx]
        return X, y, idx
