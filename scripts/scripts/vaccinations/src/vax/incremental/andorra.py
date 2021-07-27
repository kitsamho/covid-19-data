from datetime import datetime
import re

import pandas as pd

from vax.utils.incremental import enrich_data, increment, clean_count
from vax.utils.utils import get_soup
from vax.utils.dates import clean_date


class Andorra:
    def __init__(self, source_url: str, location: str):
        self.source_url = source_url
        self.location = location

    def read(self) -> pd.Series:
        soup = get_soup(self.source_url, verify=False)
        return self.parse_data(soup)

    def parse_data(self, soup):
        regex = (
            r"s’han administrat un total de ([\d\.]+) vacunes, ([\d\.]+) persones (?:han rebut|tenen) una dosi del "
            r"vaccí,?.* i ([\d\.]+) (persones )?(en )?tenen les dues"
        )
        match = re.search(regex, soup.text)
        # Metrics
        total_vaccinations = clean_count(match.group(1))
        people_vaccinated = clean_count(match.group(2))
        people_fully_vaccinated = clean_count(match.group(3))
        # people_fully_vaccinated = total_vaccinations - people_vaccinated
        return pd.Series(
            {
                "total_vaccinations": total_vaccinations,
                "people_vaccinated": people_vaccinated,
                "people_fully_vaccinated": people_fully_vaccinated,
                "date": self.parse_date(soup),
            }
        )

    def parse_date(self, soup):
        h4 = soup.find("h4", text=re.compile(r"Actualització \d{1,2}.\d{1,2}.\d{4}"))
        return clean_date(h4.text, "Actualització %d.%m.%Y")

    def pipe_location(self, ds: pd.Series) -> pd.Series:
        return enrich_data(ds, "location", self.location)

    def pipe_vaccine(self, ds: pd.Series) -> pd.Series:
        return enrich_data(ds, "vaccine", "Oxford/AstraZeneca, Pfizer/BioNTech")

    def pipe_source(self, ds: pd.Series) -> pd.Series:
        return enrich_data(ds, "source_url", self.source_url)

    def pipeline(self, ds: pd.Series) -> pd.Series:
        return (
            ds.pipe(self.pipe_location).pipe(self.pipe_vaccine).pipe(self.pipe_source)
        )

    def to_csv(self, paths):
        """Generalized."""
        data = self.read().pipe(self.pipeline)
        increment(
            paths=paths,
            location=data["location"],
            total_vaccinations=data["total_vaccinations"],
            people_vaccinated=data["people_vaccinated"],
            people_fully_vaccinated=data["people_fully_vaccinated"],
            date=data["date"],
            source_url=data["source_url"],
            vaccine=data["vaccine"],
        )


def main(paths):
    Andorra(
        source_url="https://www.govern.ad/covid19_newsletter/",
        location="Andorra",
    ).to_csv(paths)


if __name__ == "__main__":
    main()
