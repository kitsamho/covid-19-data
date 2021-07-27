import pandas as pd


def read(source_url):
    return pd.read_csv(
        source_url,
        usecols=["datum", "pocet_PCR_testy", "pocet_AG_testy", "incidence_pozitivni"],
    )


def pipeline(df: pd.DataFrame, location: str) -> pd.DataFrame:
    # Rename
    df = df.rename(
        columns={
            "datum": "Date",
            "pocet_PCR_testy": "pcr",
            "pocet_AG_testy": "antigen",
            "incidence_pozitivni": "positive",
        }
    )
    # New columns
    df = df.assign(
        **{
            "Daily change in cumulative total": df.pcr + df.antigen,
            "Country": location,
            "Units": "tests performed",
            "Source URL": "https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19",
            "Source label": "Ministry of Health",
            "Notes": pd.NA,
            "Cumulative total": pd.NA,
        }
    )
    df = df.assign(
        **{
            "Positive rate": (
                (
                    df["positive"].rolling(7).sum()
                    / df["Daily change in cumulative total"].rolling(7).sum()
                ).round(3)
            )
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
            "Cumulative total",
            "Country",
            "Units",
            "Source URL",
            "Source label",
            "Notes",
        ]
    ]
    return df


def main():
    source_url = (
        "https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/testy-pcr-antigenni.csv"
    )
    location = "Czechia"
    df = read(source_url).pipe(pipeline, location)
    df.to_csv(f"automated_sheets/{location}.csv", index=False)


if __name__ == "__main__":
    main()
