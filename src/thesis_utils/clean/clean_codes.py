from pandas import DataFrame

from thesis_utils.constants import Constants


# TODO: Add sanity checks
# TODO: Log DEBUG
def clean_codes(data: DataFrame, excluded_countries=Constants.EXCLUDED_COUNTRY_CODES):
    data = data[~data.isin(excluded_countries).any(axis=1)]
    data = data.rename(columns={"CountryCode": "UNDS"})
    return data
