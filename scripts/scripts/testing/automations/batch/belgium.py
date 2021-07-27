import pandas as pd


def read(source_url):
    # Read
    return pd.read_csv(source_url, usecols=["DATE", "TESTS_ALL", "TESTS_ALL_POS"])


def pipeline(df: pd.DataFrame, source_url: str, location: str) -> pd.DataFrame:
    df = df.groupby("DATE", as_index=False).sum()
    # Positive rate
    df = df.assign(
        **{
            "Positive rate": (
                df.TESTS_ALL_POS.rolling(7).sum() / df.TESTS_ALL.rolling(7).sum()
            ).round(3)
        }
    )
    # Rename columns
    df = df.rename(
        columns={
            "DATE": "Date",
            "TESTS_ALL": "Daily change in cumulative total",
            "PR": "Positive rate",
        }
    )
    # Add columns
    df = df.assign(
        **{
            "Country": location,
            "Units": "tests performed",
            "Source URL": source_url,
            "Source label": "Sciensano (Belgian institute for health)",
            "Notes": pd.NA,
        }
    )
    # Order
    df = df.sort_values("Date")
    # Output
    df = df[
        [
            "Date",
            "Daily change in cumulative total",
            "Positive rate",
            "Country",
            "Units",
            "Source URL",
            "Source label",
            "Notes",
        ]
    ]
    return df


def main():
    source_url = "https://epistat.sciensano.be/Data/COVID19BE_tests.csv"
    location = "Belgium"
    df = read(source_url).pipe(pipeline, source_url, location)
    df.to_csv(f"automated_sheets/{location}.csv", index=False)


if __name__ == "__main__":
    main()
