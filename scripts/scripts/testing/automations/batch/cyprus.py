from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
import requests


class Cyprus:
    def __init__(self):
        self.source_url = "https://www.data.gov.cy/node/4617?language=en"
        self.location = "Cyprus"

    def read(self):
        soup = BeautifulSoup(requests.get(self.source_url).content, "html.parser")
        url = soup.find_all(class_="data-link")[-1]["href"]
        df = pd.read_csv(url, usecols=["date", "total tests"])
        return df

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rename
        df = df.rename(
            columns={
                "date": "Date",
                "total tests": "Cumulative total",
            }
        )
        # Remove NaNs
        df = df[~df["Cumulative total"].isna()]
        # Date
        df = df.assign(
            **{
                "Date": df.Date.apply(
                    lambda x: datetime.strptime(x, "%d/%m/%Y").strftime("%Y-%m-%d")
                ),
                "Country": self.location,
                "Source label": "Ministry of Health",
                "Source URL": self.source_url,
                "Units": "tests performed",
                "Notes": pd.NA,
            }
        )
        return df

    def merge_with_current_data(self, df: pd.DataFrame, filepath: str) -> pd.DataFrame:
        df_current = pd.read_csv(filepath)
        df_current = df_current[df_current.Date < df.Date.min()]
        df = pd.concat([df_current, df]).sort_values("Date")
        return df

    def to_csv(self):
        output_path = f"automated_sheets/{self.location}.csv"
        df = (
            self.read()
            .pipe(self.pipeline)
            .pipe(self.merge_with_current_data, output_path)
        )
        df.to_csv(output_path, index=False)


def main():
    Cyprus().to_csv()


if __name__ == "__main__":
    main()
