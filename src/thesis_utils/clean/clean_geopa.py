from geopandas import GeoDataFrame
from pandas import DataFrame

from thesis_utils.constants import Constants


def clean_geopa(
    data: DataFrame | GeoDataFrame, intersection_countries=None, drop_intersection=True
) -> GeoDataFrame:
    data = data.drop(data.loc[data["ADMIN"] == "Antarctica"].index)
    for key in Constants.REPLACE_COUNTRY.keys():
        for value in Constants.REPLACE_COUNTRY[key]["values"]:
            data = data.replace(value, key)

    data = data.rename(columns={"ADM0_A3": "ISO3"})

    if drop_intersection:
        data = data.drop(data.loc[~data["ISO3"].isin(intersection_countries)].index)
    return data
