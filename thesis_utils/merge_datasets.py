import pandas as pa
from numpy import ndarray
from pandas import DataFrame

import thesis_utils as tu


def merge_datasets(
    codes: DataFrame,
    intersection_labels: ndarray,
    gdp: DataFrame,
    dist: DataFrame,
    gsdb: DataFrame,
    records: DataFrame,
) -> DataFrame:
  # 1. Create pairs from intersection labels for each country
  codes_intersect = codes[codes["ISO3"].isin(intersection_labels)].reset_index(
    drop=True
  )
  codes_pairs = codes_intersect.merge(
    codes_intersect, how="cross", suffixes=("_reporter", "_partner")
  )
  codes_pairs = codes_pairs.rename(
    columns={
      "Country Name_partner": "CNAME_partner",
      "Country Name_reporter": "CNAME_reporter",
    }
  )

  # 2. Create pairs with country pairs and years from `tu.Constants.START_YEAR` to
  # `tu.Constants.END_YEAR` (1988-2023 unless specified otherwise)
  years = { "Year": range(tu.Constants.START_YEAR, tu.Constants.END_YEAR + 1) }
  years = pa.DataFrame(data=years)
  codes_pairs_years = codes_pairs.merge(years, how="cross")

  # 3. Pivot GDP to long format
  gdp_filter = gdp[gdp["ISO3"].isin(intersection_labels)].reset_index(drop=True)
  gdp_filter = gdp_filter.rename(columns={ "Value": "GDP" })
  gdp_df_long = (
    (
      gdp_filter.melt(id_vars="ISO3", var_name="Year", value_name="GDP").assign(
        Year=lambda d: d["Year"].astype(int)
      )
    ).sort_values(["ISO3", "Year"])
  ).reset_index(drop=True)

  # 4. Merge for each pair of countries the GDP of reporter and partner
  codes_pairs_years_m1 = (
    codes_pairs_years.merge(
      gdp_df_long,
      how="inner",
      left_on=["ISO3_reporter", "Year"],
      right_on=["ISO3", "Year"],
    ).drop(columns="ISO3")
  ).rename(columns={ "GDP": "GDP_reporter" })
  codes_pairs_years_final = (
    codes_pairs_years_m1.merge(
      gdp_df_long,
      how="inner",
      left_on=["ISO3_partner", "Year"],
      right_on=["ISO3", "Year"],
    ).drop(columns="ISO3")
  ).rename(columns={ "GDP": "GDP_partner" })
  codes_pairs_years_final = codes_pairs_years_final[
    codes_pairs_years_final["ISO3_partner"]
    != codes_pairs_years_final["ISO3_reporter"]
    ].reset_index(drop=True)

  # 5. Add distances to dataset
  dist_merge = dist.drop(columns=["dist", "distwces"])
  codes_pairs_years_dist = codes_pairs_years_final.merge(
    dist_merge,
    how="inner",
    left_on=["ISO3_reporter", "ISO3_partner"],
    right_on=["origin", "destination"],
  ).drop(columns=["origin", "destination"])

  # 6. For each reporter country, load the IMPORT_YEAR and EXPORT_YEAR for that country and add columns to dataset
  # for each year. Before merging the datasets add empty columns for IMPORT, EXPORT

  # Create a dictionary for mapping Country Name -> ISO3
  codes_pairs_years_dist_trade = codes_pairs_years_dist.copy(deep=True)
  codes_pairs_years_dist_trade = codes_pairs_years_dist_trade.merge(
    records, on=["ISO3_reporter", "ISO3_partner", "Year"], how="left"
  )

  # Fill NAs with 0 for IMPORT and EXPORT column. (Temporary solution, is there a better way to impute data or
  # should I even impute data?)
  codes_pairs_years_dist_trade["IMPORT"] = codes_pairs_years_dist_trade[
    "IMPORT"
  ].fillna(value=0)
  codes_pairs_years_dist_trade["EXPORT"] = codes_pairs_years_dist_trade[
    "EXPORT"
  ].fillna(value=0)

  # 7. Match each sanction pair to the year and country pair. Pairs that do not have any active sanctions in
  # that year should have all dummy variables set to 0 and descriptions to empty strings.

  gsdb_merge = gsdb.drop(
    columns=[
      "case_id",
      "objective",
      "success",
      "sanctioning_state",
      "sanctioned_state",
    ]
  )
  gsdb_merge = gsdb_merge.rename(
    columns={
      "ISO3": "ISO3_reporter",
      "sanctioned_state_iso3": "ISO3_partner",
      "year": "Year",
    }
  )

  codes_pairs_years_dist_trade_sanctions = codes_pairs_years_dist_trade.merge(
    gsdb_merge, on=["ISO3_reporter", "ISO3_partner", "Year"], how="left"
  )
  codes_pairs_years_dist_trade_sanctions["arms"] = (
    codes_pairs_years_dist_trade_sanctions["arms"].fillna(value=0)
  )
  codes_pairs_years_dist_trade_sanctions["military"] = (
    codes_pairs_years_dist_trade_sanctions["military"].fillna(value=0)
  )
  codes_pairs_years_dist_trade_sanctions["trade"] = (
    codes_pairs_years_dist_trade_sanctions["trade"].fillna(value=0)
  )
  codes_pairs_years_dist_trade_sanctions["descr_trade"] = (
    codes_pairs_years_dist_trade_sanctions["descr_trade"].fillna(value="")
  )
  codes_pairs_years_dist_trade_sanctions["financial"] = (
    codes_pairs_years_dist_trade_sanctions["financial"].fillna(value=0)
  )
  codes_pairs_years_dist_trade_sanctions["travel"] = (
    codes_pairs_years_dist_trade_sanctions["travel"].fillna(value=0)
  )
  codes_pairs_years_dist_trade_sanctions["other"] = (
    codes_pairs_years_dist_trade_sanctions["other"].fillna(value=0)
  )
  codes_pairs_years_dist_trade_sanctions["target_mult"] = (
    codes_pairs_years_dist_trade_sanctions["target_mult"].fillna(value=0)
  )
  codes_pairs_years_dist_trade_sanctions["sender_mult"] = (
    codes_pairs_years_dist_trade_sanctions["sender_mult"].fillna(value=0)
  )

  return codes_pairs_years_dist_trade_sanctions
