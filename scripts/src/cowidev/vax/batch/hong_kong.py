import requests
import pandas as pd
from cowidev.vax.utils.files import export_metadata


def read(source: str) -> pd.DataFrame:
    data = requests.get(source).json()
    return pd.DataFrame.from_dict(
        [
            {
                "date": d["date"],
                "people_vaccinated": d["firstDose"]["cumulative"]["total"],
                "people_fully_vaccinated": d["secondDose"]["cumulative"]["total"],
                "total_vaccinations": d["totalDose"]["cumulative"]["total"],
                "total_pfizer": d["totalDose"]["cumulative"]["biontech"],
                "total_sinovac": d["totalDose"]["cumulative"]["sinovac"],
            }
            for d in data
        ]
    )


def enrich_vaccine(df: pd.DataFrame) -> pd.DataFrame:
    def _enrich_vaccine(date: str) -> str:
        if date < "2021-03-06":
            return "Sinovac"
        return "Pfizer/BioNTech, Sinovac"

    return df.assign(vaccine=df.date.apply(_enrich_vaccine))


def enrich_metadata(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        location="Hong Kong", source_url="https://www.covidvaccine.gov.hk/en/dashboard"
    )


def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    return df.pipe(enrich_vaccine).pipe(enrich_metadata)


def manufacturer_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["location", "date", "total_pfizer", "total_sinovac"]]
        .rename(columns={"total_pfizer": "Pfizer/BioNTech", "total_sinovac": "Sinovac"})
        .set_index(["location", "date"])
        .stack()
        .reset_index()
        .rename(columns={"level_2": "vaccine", 0: "total_vaccinations"})
    )


def main(paths):
    source = "https://static.data.gov.hk/covid-vaccine/bar_vaccination_date.json"
    data = read(source).pipe(pipeline)

    destination = paths.tmp_vax_out("Hong Kong")
    data.drop(columns=["total_pfizer", "total_sinovac"]).to_csv(
        destination, index=False
    )

    destination = paths.tmp_vax_out_man("Hong Kong")
    manufacturer = data.pipe(manufacturer_pipeline)
    manufacturer.to_csv(destination, index=False)
    export_metadata(
        manufacturer, "Government of Hong Kong", source, paths.tmp_vax_metadata_man
    )


if __name__ == "__main__":
    main()
