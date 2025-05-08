import pandas as pa
from numpy import ndarray
from pandas import DataFrame


def toIso3(data: DataFrame, iso_lookup) -> DataFrame:
    data["ISO3_reporter"] = data["Reporter Name"].str.strip().map(iso_lookup)
    data["ISO3_partner"] = data["Partner Name"].str.strip().map(iso_lookup)
    return data


def clean_trade(intersection_labels: ndarray, codes: DataFrame):
    IMPORT_PATH = "../data/trade/import/IMPORT_"
    EXPORT_PATH = "../data/trade/export/EXPORT_"
    EXTENSION = ".xlsx"

    iso_lookup = codes.set_index("Country Name")["ISO3"].to_dict()
    records = pa.DataFrame(
        columns=["ISO3_reporter", "ISO3_partner", "Year", "IMPORT", "EXPORT"]
    )
    reports = []

    for label in intersection_labels:

        # Paths to files
        file_path_import = IMPORT_PATH + label + EXTENSION
        file_path_export = EXPORT_PATH + label + EXTENSION

        # Load files
        import_file = pa.read_excel(file_path_import).drop(
            columns=["Trade Flow", "Product Group", "Indicator"]
        )
        export_file = pa.read_excel(file_path_export).drop(
            columns=["Trade Flow", "Product Group", "Indicator"]
        )

        # Convert partner and reporter name to ISO3, drop any NA on ISO3_partner
        import_file = (toIso3(import_file, iso_lookup)).dropna(
            axis="index", subset=["ISO3_partner", "ISO3_reporter"]
        )
        export_file = (toIso3(export_file, iso_lookup)).dropna(
            axis="index", subset=["ISO3_partner", "ISO3_reporter"]
        )

        # Convert to long format
        import_long = (
            (
                (
                    import_file.melt(
                        id_vars=[
                            "Reporter Name",
                            "Partner Name",
                            "ISO3_partner",
                            "ISO3_reporter",
                        ],
                        var_name="Year",
                        value_name="IMPORT",
                    )
                ).assign(Year=lambda d: d["Year"].astype(int))
            ).sort_values(["ISO3_reporter", "Year"])
        ).reset_index(drop=True)
        export_long = (
            (
                (
                    export_file.melt(
                        id_vars=[
                            "Reporter Name",
                            "Partner Name",
                            "ISO3_partner",
                            "ISO3_reporter",
                        ],
                        var_name="Year",
                        value_name="EXPORT",
                    )
                ).assign(Year=lambda d: d["Year"].astype(int))
            ).sort_values(["ISO3_reporter", "Year"])
        ).reset_index(drop=True)
        import_long = import_long.drop(columns=["Reporter Name", "Partner Name"])
        export_long = export_long.drop(columns=["Reporter Name", "Partner Name"])
        merged_import_export = import_long.merge(
            export_long, on=["ISO3_partner", "ISO3_reporter", "Year"], how="inner"
        )
        if not merged_import_export.empty:
            reports.append(merged_import_export)

    if reports:
        records = pa.concat(reports, ignore_index=True)
        # Imputation of IMPORT and EXPORT, not needed as PPML takes those into account
        # for col in ["IMPORT", "EXPORT"]:
        #     records[col] = (
        #         records.sort_values(by=["ISO3_reporter", "ISO3_partner", "Year"])
        #         .groupby(["ISO3_reporter", "ISO3_partner"])[col]
        #         .transform(
        #             lambda x: x.interpolate(
        #                 method="linear", limit_direction="both"
        #             ).fillna(x.rolling(window=20, min_periods=1).median())
        #         )
        #     )
        # for col in ["IMPORT", "EXPORT"]:
        #     records[col] = records[col].mask(records[col] < 1, 0)
        return records
    else:
        return None
