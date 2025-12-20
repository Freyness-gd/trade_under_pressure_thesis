import pandas as pd
from pandas import DataFrame

from thesis_utils.constants import Constants


# TODO: Add sanity checks
# TODO Log DEBUG
def clean_cepii(
    data: DataFrame, excluded_countries=Constants.EXCLUDED_COUNTRY_CODES
) -> DataFrame:
    # na_rows = data[data.isna().any(axis=1)][["origin", "destination"]]
    # # Sanity check
    # na_rows.isna().sum()
    # print("Percentage of NaN rows: ", (na_rows.shape[0] / data.shape[0]) * 100, "%")
    data = data.drop(["comcol", "curcol", "col45"], axis=1)
    data = data.rename(columns={"iso_o": "origin", "iso_d": "destination"})
    data = data.dropna()
    # # Sanity check
    # data.isna().any()
    # print("Unique countries in origin column", data["origin"].nunique())
    # print("Unique countries in destination column", data["destination"].nunique())

    for key in Constants.REPLACE_COUNTRY.keys():
        for value in Constants.REPLACE_COUNTRY[key]["values"]:
            data = data.replace(value, key)

    data = data[~data.isin(excluded_countries).any(axis=1)]

    # Set distance for SER to the distance of YUG
    yug_rows = data[
        data.apply(lambda row: row.astype(str).str.contains("YUG").any(), axis=1)
    ].copy()
    yug_rows = yug_rows.replace("YUG", "SER")
    data = pd.concat([data, yug_rows], ignore_index=True)

    return data
