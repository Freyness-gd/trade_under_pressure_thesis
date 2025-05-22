from typing import Sequence, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LaggedDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        features: Sequence[str],
        target: str,
        lag: int = 0,
        lag_columns: Sequence[str] | None = None,
    ):
        if lag_columns is None:
            lag_columns = []

        df = data.copy()

        for col in lag_columns:
            for i in range(1, lag + 1):
                df[f"{col}_lag{i}"] = df.groupby("dyad_id", sort=False)[col].shift(i)

        new_lag_cols = [f"{c}_lag{i}" for c in lag_columns for i in range(1, lag + 1)]
        df = df.dropna(subset=new_lag_cols).reset_index(drop=True)

        self.feature_cols: List[str] = list(features) + new_lag_cols
        self.X = torch.tensor(df[self.feature_cols].to_numpy(np.float32))
        self.y = torch.tensor(df[target].to_numpy(np.float32)).unsqueeze(1)

        self.row_idx = torch.tensor(df.index.to_numpy())

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx].unsqueeze(0), self.y[idx], int(self.row_idx[idx])
