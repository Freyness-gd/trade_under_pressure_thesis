import numpy as np
import torch
from numpy.lib._stride_tricks_impl import sliding_window_view
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    def __init__(self, data, features, target, seq_len: int = 5, horizon: int = 1):
        self.X_list, self.y_list = [], []
        min_len = seq_len + horizon

        for _, grp in data.groupby("dyad_id", sort=False):
            if len(grp) < min_len:
                continue

            feats = grp[features].to_numpy(np.float32)
            tgt = grp[target].to_numpy(np.float32)

            feat_win = sliding_window_view(feats, seq_len, axis=0)

            tgt_win_all = sliding_window_view(tgt, horizon, axis=0)
            tgt_win = tgt_win_all[seq_len:]

            feat_win = feat_win[: len(tgt_win)]

            self.X_list.append(torch.from_numpy(feat_win.copy()))
            self.y_list.append(torch.from_numpy(tgt_win.copy()))

        if not self.X_list:
            self.X = torch.empty((0, seq_len, len(features)))
            self.y = torch.empty((0, horizon))
            return

        self.X = torch.cat(self.X_list)
        self.y = torch.cat(self.y_list)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].float().transpose(0, 1), self.y[idx], idx
