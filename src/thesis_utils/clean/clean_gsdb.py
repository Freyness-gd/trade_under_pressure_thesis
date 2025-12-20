import pandas as pa
from pandas import DataFrame

from thesis_utils.constants import Constants


# TODO: Log with DEBUG
# TODO: Add sanity checks
def clean_gsdb(
    data: DataFrame,
    start_year=Constants.START_YEAR,
    end_year=Constants.END_YEAR,
    excluded_countries=Constants.EXCLUDED_COUNTRY_CODES,
):
    if start_year < Constants.START_YEAR or end_year > Constants.END_YEAR:
        raise IndexError("Start or End Year out of bounds")

    # Remove sanctions against terrorist organisations
    data = data[data["sanctioned_state_iso3"].astype(str) != ""]

    for key in Constants.REPLACE_COUNTRY.keys():
        for value in Constants.REPLACE_COUNTRY[key]["values"]:
            data = data.replace(value, key)

    data = filter_excluded_countries(data, excluded_countries)

    # Sanity checks
    # (data["sanctioned_state_iso3"].astype(str) == '').sum()
    # data.isna().any()

    data = data.rename(columns={"sanctioning_state_iso3": "ISO3"})

    data["year"] = pa.to_datetime(data["year"]).dt.year

    return data


def filter_excluded_countries(
    data: DataFrame,
    excluded_countries=Constants.EXCLUDED_COUNTRY_CODES,
):
    data = data[~data.isin(excluded_countries).any(axis=1)]
    return data
