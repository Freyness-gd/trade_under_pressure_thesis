import pandas as pd
import polars as pl


def add_fixed_effects(df: pd.DataFrame) -> pd.DataFrame:
    pl_df = pl.DataFrame(
        {
            "fe_dyad_id": df["dyad_id"],
        }
    )

    df_out = df.copy(deep=True)
    print("Df Shape: ", df_out.shape)
    fe_dummies = pl_df.to_dummies(columns=["fe_dyad_id"], drop_first=True).to_pandas()
    print("Dummies Shape: ", fe_dummies.shape)

    df_out = pd.concat(
        [df.reset_index(drop=True), fe_dummies.reset_index(drop=True)], axis=1
    )
    print("Final Df Shape: ", df_out.shape)

    return df_out
