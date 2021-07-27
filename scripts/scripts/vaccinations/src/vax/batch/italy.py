import pandas as pd

from vax.utils.files import export_metadata


def main(paths):

    vaccine_mapping = {
        "Pfizer/BioNTech": "Pfizer/BioNTech",
        "Moderna": "Moderna",
        "Vaxzevria (AstraZeneca)": "Oxford/AstraZeneca",
        "Janssen": "Johnson&Johnson",
    }
    one_dose_vaccines = ["Johnson&Johnson"]

    url = (
        "https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/"
        "somministrazioni-vaccini-latest.csv"
    )
    df = pd.read_csv(
        url,
        usecols=[
            "data_somministrazione",
            "fornitore",
            "fascia_anagrafica",
            "prima_dose",
            "seconda_dose",
            "pregressa_infezione",
        ],
    )
    assert set(df["fornitore"].unique()) == set(vaccine_mapping.keys())
    df = df.replace(vaccine_mapping)
    df["total_vaccinations"] = (
        df["prima_dose"] + df["seconda_dose"] + df["pregressa_infezione"]
    )
    df["people_vaccinated"] = df["prima_dose"] + df["pregressa_infezione"]
    df = df.rename(
        columns={
            "data_somministrazione": "date",
            "fornitore": "vaccine",
            "fascia_anagrafica": "age_group",
        }
    )
    # df_age_group = df.copy()

    # Data by manufacturer
    by_manufacturer = (
        df.groupby(["date", "vaccine"], as_index=False)["total_vaccinations"]
        .sum()
        .sort_values("date")
    )
    by_manufacturer["total_vaccinations"] = by_manufacturer.groupby("vaccine")[
        "total_vaccinations"
    ].cumsum()
    by_manufacturer["location"] = "Italy"
    by_manufacturer.to_csv(paths.tmp_vax_out_man("Italy"), index=False)
    export_metadata(
        by_manufacturer,
        "Extraordinary commissioner for the Covid-19 emergency",
        url,
        paths.tmp_vax_metadata_man,
    )

    # Vaccination data
    df = df.rename(
        columns={
            "seconda_dose": "people_fully_vaccinated",
        }
    )
    df.loc[
        df.vaccine.isin(one_dose_vaccines), "people_fully_vaccinated"
    ] = df.people_vaccinated
    df = (
        df.groupby("date", as_index=False)[
            ["total_vaccinations", "people_vaccinated", "people_fully_vaccinated"]
        ]
        .sum()
        .sort_values("date")
    )

    df[["total_vaccinations", "people_vaccinated", "people_fully_vaccinated"]] = df[
        ["total_vaccinations", "people_vaccinated", "people_fully_vaccinated"]
    ].cumsum()

    df.loc[:, "location"] = "Italy"
    df.loc[:, "source_url"] = url
    df.loc[:, "vaccine"] = ", ".join(sorted(vaccine_mapping.values()))

    df.to_csv(paths.tmp_vax_out("Italy"), index=False)

    # Vaccination Age Group data
    # by_age_group = (
    #     df_age_group.groupby(["date", "age_group"], as_index=False)["total_vaccinations"]
    #         .sum()
    #         .sort_values("date")
    # )

    # by_age_group["total_vaccinations"] = by_age_group.groupby("age_group")["total_vaccinations"].cumsum()
    # by_age_group[["age_group_min", "age_group_max"]] = by_age_group.age_group.apply(
    #     lambda x: pd.Series(str(x).split("-")))
    # by_age_group["age_group_min"] = by_age_group["age_group_min"].str.replace("+", "", regex=False)
    # by_age_group.drop(columns=["age_group"])
    # by_age_group["location"] = "Italy"
    # by_age_group = by_age_group[["date", "age_group_min", "age_group_max", "total_vaccinations", "location"]]

    # by_age_group.to_csv(paths.tmp_vax_out_by_age_group("Italy"), index=False)


if __name__ == "__main__":
    main()
