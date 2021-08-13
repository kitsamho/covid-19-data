from datetime import datetime

import pandas as pd


class Grapheriser:
    def __init__(
        self,
        location: str = "location",
        date: str = "date",
        date_ref: datetime = datetime(2020, 1, 21),
        fillna: bool = False,
        fillna_0: bool = True,
        pivot_column: str = None,
        pivot_values: str = None,
    ) -> None:
        self.location = location
        self.date = date
        self.date_ref = date_ref
        self.fillna = fillna
        self.fillna_0 = fillna_0
        self.pivot_column = pivot_column
        self.pivot_values = pivot_values

    @property
    def columns_metadata(self) -> list:
        return ["Country", "Year"]

    def columns_data(self, df: pd.DataFrame) -> list:
        return [col for col in df.columns if col not in self.columns_metadata]

    def pipe_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pivot_column is not None and self.pivot_values is not None:
            return df.pivot(
                index=[self.location, self.date],
                columns=self.pivot_column,
                values=self.pivot_values,
            ).reset_index()
        return df

    def pipe_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = (
            df.rename(
                columns={
                    self.location: "Country",
                }
            )
            .assign(date=(df[self.date] - self.date_ref).dt.days)
            .rename(columns={"date": "Year"})
        ).copy()
        return df

    def pipe_order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        col_order = self.columns_metadata + self.columns_data(df)
        df = df[col_order].sort_values(col_order)
        return df

    def pipe_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_data = self.columns_data(df)
        if self.fillna:
            df[columns_data] = df.groupby(["Country"])[columns_data].fillna(
                method="ffill"
            )
        if self.fillna_0:
            df[columns_data] = df[columns_data].fillna(0)
        return df

    def pipeline(self, input_path: str):
        df = pd.read_csv(input_path, parse_dates=[self.date])
        df = (
            df.pipe(self.pipe_pivot)
            .pipe(self.pipe_metadata_columns)
            .pipe(self.pipe_order_columns)
            .pipe(self.pipe_fillna)
        )
        return df

    def run(self, input_path: str, output_path: str):
        df = self.pipeline(input_path)
        df.to_csv(output_path, index=False)
