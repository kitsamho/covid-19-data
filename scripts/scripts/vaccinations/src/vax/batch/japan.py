from datetime import datetime

import re
import requests
import pandas as pd

from vax.utils.utils import get_soup, clean_df_columns_multiindex
from vax.utils.dates import clean_date_series
from vax.utils.files import export_metadata


class Japan:
    def __init__(self):
        self.source_url_1 = "https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/vaccine_sesshujisseki.html"
        self._base_url_2 = "https://www.kantei.go.jp/"
        self.source_sheet_name_health = "医療従事者等"
        self.source_sheet_name_general = "一般接種"
        self.location = "Japan"
        self.vaccine_mapping = {
            "ファイザー社": "Pfizer/BioNTech",
            "武田/モデルナ社": "Moderna",
        }

    @property
    def source_url_2(self):
        response = requests.get(self.source_url_2_ref)
        response.encoding = "utf-8"
        source_2_path = re.search('日別の実績.*?href="(.*?\\.xlsx)"', response.text).group(1)
        return self._base_url_2 + source_2_path

    @property
    def source_url_2_ref(self):
        return self._base_url_2 + "jp/headline/kansensho/vaccine.html"

    def read(self):
        df_1 = self.read_1().pipe(self.pipeline_1)
        df_2 = self.read_2().pipe(self.pipeline_2)
        return pd.concat([df_1, df_2]).reset_index(drop=True)

    def read_1(self):
        soup = get_soup(self.source_url_1)
        dfs = pd.read_html(str(soup), header=0)
        if len(dfs) != 1:
            raise ValueError(
                f"Only one table should be present. {len(dfs)} tables detected."
            )
        df = dfs[0]
        return df

    def pipe_1_rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(
            columns={
                "日付": "date",
                "接種回数": "total_vaccinations",
                "内１回目": "first_dose",
                "内２回目": "second_dose",
            }
        )

    def pipe_1_filter_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df.date != "合計"]

    def pipe_1_source(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(source_url=self.source_url_1)

    def pipe_1_vaccine(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(vaccine="Pfizer/BioNTech")

    def pipe_1_date(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(date=clean_date_series(df.date, "%Y/%m/%d"))

    def pipeline_1(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.pipe(self.pipe_1_rename_columns)
            .pipe(self.pipe_1_filter_dates)
            .pipe(self.pipe_1_source)
            .pipe(self.pipe_1_vaccine)
            .pipe(self.pipe_1_date)
        )

    def read_2(self):
        df_health = self._parse_data_2(
            source=self.source_url_2,
            sheet_name=self.source_sheet_name_health,
            header=[2, 3],
            date_col="集計日",
        )
        df_general = self._parse_data_2(
            source=self.source_url_2,
            sheet_name=self.source_sheet_name_general,
            header=[2, 3, 4],
            date_col="接種日",
            extra_levels=["すべて"],
        )
        return pd.concat(
            [
                df_health,
                df_general,
            ]
        ).reset_index(drop=True)

    def _parse_data_2(
        self,
        source: str,
        sheet_name: str,
        header: list,
        date_col: str,
        extra_levels: list = None,
    ) -> pd.DataFrame:
        doses_1, doses_2 = self._parse_data_2_doses(
            source, sheet_name, header, date_col, extra_levels
        )
        doses_1 = (
            doses_1.stack()
            .reset_index()
            .rename(columns={date_col: "date", "level_1": "vaccine", 0: "first_dose"})
        )
        doses_2 = (
            doses_2.stack()
            .reset_index()
            .rename(columns={date_col: "date", "level_1": "vaccine", 0: "second_dose"})
        )
        df = doses_1.merge(doses_2, on=["date", "vaccine"])
        return df

    def _parse_data_2_doses(
        self,
        source: str,
        sheet_name: str,
        header: list,
        date_col: str,
        extra_levels: list = None,
    ) -> pd.DataFrame:
        # Read general
        df = pd.read_excel(source, sheet_name=sheet_name, header=header)
        df = df.dropna(axis=1, how="all")
        # Clean column names
        df = clean_df_columns_multiindex(df)
        # Filter date rows
        df = (
            df[df[date_col].apply(isinstance, args=(datetime,))]
            .set_index(date_col)
            .sort_index()
        )
        # Build DataFrame
        if extra_levels is None:
            extra_levels = []
        doses_1 = df.loc[:, tuple(extra_levels + ["内1回目"])]
        doses_2 = df.loc[:, tuple(extra_levels + ["内2回目"])]
        return doses_1, doses_2

    def _parse_data_2_vaccines(self, doses_1, doses_2, date_col) -> pd.DataFrame:
        if not all(doses_1.columns == doses_2.columns):
            raise ValueError("Missmatch in vaccines for dose 1 and dose 2")
        x = doses_1.cumsum()
        vaccines_wrong = set(x.columns).difference(self.vaccine_mapping)
        if vaccines_wrong:
            raise ValueError(f"Unknown vaccine(s): {vaccines_wrong}")
        vaccines = (
            x.mask(x == 0).stack().reset_index().groupby(date_col).level_1.unique()
        )
        return vaccines

    def pipe_2_aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(["date", "vaccine"], as_index=False).agg(
            {
                "first_dose": "sum",
                "second_dose": "sum",
            }
        )

    def pipe_2_vaccine_rename(self, df: pd.DataFrame) -> pd.DataFrame:
        vaccines_wrong = set(df.vaccine).difference(self.vaccine_mapping)
        if vaccines_wrong:
            raise ValueError(f"Unknown vacine(s): {vaccines_wrong}")
        return df.assign(vaccine=df.vaccine.replace(self.vaccine_mapping))

    def pipe_2_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(total_vaccinations=df.first_dose + df.second_dose)

    def pipe_2_source(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(source_url=self.source_url_2_ref)

    def pipe_2_date(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(date=clean_date_series(df.date))

    def pipeline_2(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.pipe(self.pipe_2_aggregate)
            .pipe(self.pipe_2_vaccine_rename)
            .pipe(self.pipe_2_metrics)
            .pipe(self.pipe_2_source)
            .pipe(self.pipe_2_date)
        )

    def pipe_location(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(location=self.location)

    def pipe_cumsum(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["date"])
        column_metrics = ["total_vaccinations", "first_dose", "second_dose"]
        df.loc[:, column_metrics] = df.groupby("vaccine")[column_metrics].cumsum()
        return df

    def base_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pipe(self.pipe_location).pipe(self.pipe_cumsum)[
            [
                "location",
                "date",
                "vaccine",
                "source_url",
                "total_vaccinations",
                "first_dose",
                "second_dose",
            ]
        ]

    def pipeline_manufacturer(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            [
                "location",
                "date",
                "vaccine",
                "total_vaccinations",
            ]
        ]

    def pipe_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(
            columns={
                "first_dose": "people_vaccinated",
                "second_dose": "people_fully_vaccinated",
            }
        )

    def pipe_aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(["date", "location", "source_url"], as_index=False).agg(
            {
                "total_vaccinations": sum,
                "people_vaccinated": sum,
                "people_fully_vaccinated": sum,
                "vaccine": lambda x: ", ".join(sorted(set(x))),
            }
        )

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.pipe(self.pipe_metrics).pipe(self.pipe_aggregate)[
            [
                "location",
                "date",
                "vaccine",
                "source_url",
                "total_vaccinations",
                "people_vaccinated",
                "people_fully_vaccinated",
            ]
        ]

    def export(self, paths):
        df = self.read().pipe(self.base_pipeline)
        # Drop total_vaccinations == 0 rows added by groupby.
        df = df.drop(df[df.total_vaccinations == 0].index).reset_index()
        # Manufacturer
        df.pipe(self.pipeline_manufacturer).to_csv(
            paths.tmp_vax_out_man(self.location), index=False
        )
        export_metadata(
            df,
            "Prime Minister of Japan and Hist Cabinet",
            self.source_url_2_ref,
            paths.tmp_vax_metadata_man,
        )
        # Main data
        df.pipe(self.pipeline).to_csv(paths.tmp_vax_out(self.location), index=False)


def main(paths):
    Japan().export(paths)
