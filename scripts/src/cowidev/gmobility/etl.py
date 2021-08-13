import pandas as pd


class GMobilityETL:
    def __init__(self) -> None:
        self.source_url = (
            "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
        )

    def extract(self):
        return pd.read_csv(
            self.source_url,
            usecols=[
                "country_region",
                "sub_region_1",
                "sub_region_2",
                "metro_area",
                "iso_3166_2_code",
                "census_fips_code",
                "date",
                "retail_and_recreation_percent_change_from_baseline",
                "grocery_and_pharmacy_percent_change_from_baseline",
                "parks_percent_change_from_baseline",
                "transit_stations_percent_change_from_baseline",
                "workplaces_percent_change_from_baseline",
                "residential_percent_change_from_baseline",
            ],
            low_memory=False,
        )

    def load(self, df: pd.DataFrame, output_path: str) -> None:
        # Export data
        df.to_csv(output_path, index=False)

    def run(self, output_path: str):
        df = self.extract()
        self.load(df, output_path)


def run_etl(output_path: str):
    etl = GMobilityETL()
    etl.run(output_path)
