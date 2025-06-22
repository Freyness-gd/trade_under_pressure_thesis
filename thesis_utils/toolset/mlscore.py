import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_pred - y_true))


def rmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.mean(y_true)
    return np.nan if denom == 0 else mae(y_true, y_pred) / denom


def pseudo_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sse = np.square(y_pred - y_true).sum()
    sst = np.square(y_true - y_true.mean()).sum()
    return np.nan if sst == 0 else 1.0 - sse / sst


def within_r2(y_true: np.ndarray, y_pred: np.ndarray, pair_ids: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pair_ids = np.asarray(pair_ids)

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "pair": pair_ids})
    df["y_true_c"] = df["y_true"] - df.groupby("pair")["y_true"].transform("mean")

    sse_w = np.square(df["y_pred"] - df["y_true"]).sum()
    sst_w = np.square(df["y_true_c"]).sum()
    return np.nan if sst_w == 0 else 1.0 - sse_w / sst_w


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = np.mean(np.square(y_pred - y_true))
    return np.sqrt(mse)
