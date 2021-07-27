import os
from datetime import date
import urllib.request

import pandas as pd
from bs4 import BeautifulSoup


def main():

    data = pd.read_csv("automated_sheets/Benin.csv")

    url = "https://www.gouv.bj/coronavirus/"
    req = urllib.request.urlopen(url)
    soup = BeautifulSoup(req.read(), "html.parser")

    stats = soup.find_all("h2", attrs={"class", "h1 adapt white regular"})

    count = int(stats[0].text) + int(stats[1].text)
    date_str = date.today().strftime("%Y-%m-%d")

    if count > data["Cumulative total"].max() and date_str > data["Date"].max():

        new = pd.DataFrame(
            {
                "Country": "Benin",
                "Date": [date_str],
                "Cumulative total": count,
                "Source URL": url,
                "Source label": "Government of Benin",
                "Units": "tests performed",
            }
        )

        df = pd.concat([new, data], sort=False)
        df.to_csv("automated_sheets/Benin.csv", index=False)


if __name__ == "__main__":
    main()
