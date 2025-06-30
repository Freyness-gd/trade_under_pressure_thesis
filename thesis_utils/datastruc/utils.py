from typing import Sequence, Tuple, Dict

import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset

from thesis_utils.constants import Constants
from thesis_utils.datastruc import (
    DatasetWrapper,
    DatasetWrapperOptimized,
    DatasetWrapperOptimizedWithYear,
    SlidingWindowDataset,
)
from thesis_utils.datastruc.LaggedDataset import LaggedDataset


def stratified_sample(
    data: DataFrame,
    year_col: str,
    frac: float,
    start: int,
    end: int,
    seed: int,
) -> Tuple[DataFrame, DataFrame]:
    mask = data[year_col].between(start, end)
    window = data.loc[mask]

    if frac >= 1.0:
        sampled = window.copy()
    else:
        sampled = window.groupby(year_col, group_keys=False).sample(
            frac=frac, random_state=seed
        )

    remainder = data.drop(sampled.index)

    print("Shape original data: ", data.shape)
    print("Shape sampled: ", sampled.shape)
    print("Shape remainder: ", remainder.shape)

    return sampled, remainder


def make_panel_datasets(
    data: DataFrame,
    features: Sequence[str],
    target: str,
    keep_frac: float = 1.0,
    year_col: str = "Year",
    start: int = Constants.START_YEAR,
    end: int = Constants.END_YEAR,
    horizon: int = 1,
    seed: int = 16,
) -> Tuple[Dataset, Dataset]:
    sampled_df, remainder_df = stratified_sample(
        data, year_col, keep_frac, start, end, seed
    )
    ds_sampled = DatasetWrapper(
        data=sampled_df, features=features, target=target, horizon=horizon
    )
    ds_remainder = DatasetWrapper(data=remainder_df, features=features, target=target)
    return ds_sampled, ds_remainder


def make_panel_datasets_dyad(
    data: pd.DataFrame,
    features: Sequence[str],
    target: str,
    horizon: int = 1,
) -> Tuple[Dataset, Dict[str, int]]:
    df = data.copy()
    df["dyad_idx"] = df["dyad_id"].cat.codes

    # 2) Optional: retrieve dyad → index mapping (if needed)
    dyad_to_idx = {dyad: idx for idx, dyad in enumerate(df["dyad_id"].cat.categories)}

    # 3) Wrap into Dataset
    dataset = DatasetWrapperOptimized(
        data=df,
        features=features,
        target=target,
        horizon=horizon,
    )

    return dataset, dyad_to_idx


def make_panel_datasets_dyad_year(
    data: pd.DataFrame,
    features: Sequence[str],
    target: str,
    horizon: int = 1,
) -> Tuple[Dataset, Dict[str, int]]:
    # Same as make panel datasets dyad but also return year
    df = data.copy()
    df["dyad_idx"] = df["dyad_id"].cat.codes
    df["year"] = df["Year"].astype(int)
    # 2) Optional: retrieve dyad → index mapping (if needed)
    dyad_to_idx = { dyad: idx for idx, dyad in enumerate(df["dyad_id"].cat.categories) }
    # 3) Wrap into Dataset
    dataset = DatasetWrapperOptimizedWithYear(
        data=df,
        features=features,
        target=target,
        horizon=horizon,
    )

    return dataset, dyad_to_idx


def make_panel_slidingwindows(
    data: DataFrame,
    features: Sequence[str],
    target: str,
    keep_frac: float = 1.0,
    year_col: str = "Year",
    start: int = Constants.START_YEAR,
    end: int = Constants.END_YEAR,
    seed: int = 16,
    seq_len: int = 5,
    horizon: int = 1,
):
    sampled_df, remainder_df = stratified_sample(
        data, year_col, keep_frac, start, end, seed
    )
    ds_sampled = SlidingWindowDataset(
        data=sampled_df,
        features=features,
        target=target,
        seq_len=seq_len,
        horizon=horizon,
    )
    ds_remainder = SlidingWindowDataset(
        data=remainder_df,
        features=features,
        target=target,
        seq_len=seq_len,
        horizon=horizon,
    )
    return ds_sampled, ds_remainder


def make_panel_laggedsets(
    data: DataFrame,
    features: Sequence[str],
    target: str,
    keep_frac: float = 1.0,
    year_col: str = "Year",
    start: int = Constants.START_YEAR,
    end: int = Constants.END_YEAR,
    seed: int = 16,
    lag: int = 1,
    lag_columns: Sequence[str] | None = None,
) -> Tuple[Dataset, Dataset]:
    sampled_df, remainder_df = stratified_sample(
        data, year_col, keep_frac, start, end, seed
    )
    ds_sampled = LaggedDataset(
        sampled_df, features, target, lag=lag, lag_columns=lag_columns
    )
    ds_remainder = LaggedDataset(
        remainder_df, features, target, lag=lag, lag_columns=lag_columns
    )
    return ds_sampled, ds_remainder
