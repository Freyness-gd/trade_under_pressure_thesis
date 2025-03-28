from pandas import DataFrame

import Constants


# TODO: Add sanity checks
def clean_gdp(
    data: DataFrame,
    start_year=Constants.START_YEAR,
    end_year=Constants.END_YEAR,
    missing_threshold=Constants.MISSING_THRESHOLD,
    excluded_countries=Constants.EXCLUDED_COUNTRY_CODES,
    col_first=False,
):
    if start_year < Constants.START_YEAR or end_year > Constants.END_YEAR:
        raise IndexError("Start or End Year out of bounds")

    data = data.drop(["Series Name", "Series Code", "Country Name"], axis=1)
    data = data.rename(columns=lambda x: x if not x.endswith("]") else x.split(" ")[0])
    data = data.rename(columns={"Country Code": "ISO3"})
    data = data[~data["ISO3"].isna()]

    # Include only subset of years
    data = data[["ISO3"] + [str(year) for year in range(start_year, end_year + 1)]]

    data = remove_rows_with_missing(data, missing_threshold, col_first)
    data = filter_excluded_countries(data, excluded_countries)

    data = data[~data.isin(["CHI"]).any(axis=1)]

    for key in Constants.REPLACE_COUNTRY.keys():
        for value in Constants.REPLACE_COUNTRY[key]["values"]:
            data = data.replace(value, key)

    return data


def remove_rows_with_missing(
    data: DataFrame, missing_threshold=Constants.MISSING_THRESHOLD, col_first=False
):
    # TODO: Log with DEBUG
    # print("Length before filter: ", data.shape)
    # for row in data.itertuples(name="Country"):
    #     print(
    #         data.loc[row.Index]["ISO3"],
    #         ": ",
    #         (data.loc[row.Index].isna().sum() / len(data.loc[row.Index])),
    #         "\n",
    #     )
    #     print("Len ", len(data.loc[row.Index]), "\n")
    # print("Len: ", len(data.columns))

    if col_first:
        data = data.loc[:, data.isnull().sum() / len(data) <= missing_threshold]
        data = data[
            (data.isnull().sum(axis=1) / len(data.columns)) <= missing_threshold
        ]
    else:
        data = data[
            (data.isnull().sum(axis=1) / len(data.columns)) <= missing_threshold
        ]
        data = data.loc[:, data.isnull().sum() / len(data) <= missing_threshold]

    # TODO: Log with Debug
    # print("Length after filter: ", data.shape)

    return data


def filter_excluded_countries(
    data, excluded_countries=Constants.EXCLUDED_COUNTRY_CODES
):
    data = data[~data.isin(excluded_countries).any(axis=1)]
    return data
