import os

import pandas as pd
from bs4 import BeautifulSoup
import requests
from datetime import date


def main():

    df = pd.read_csv("automated_sheets/Laos.csv")

    url = "https://www.covid19.gov.la/index.php"
    # Only works when making two requests. The first always returns an error.
    try:
        data = BeautifulSoup(requests.get(url).text, features="lxml")
    except:
        data = BeautifulSoup(requests.get(url).text, features="lxml")

    stats = data.find_all("p")
    count = int(stats[11].text.split(" ")[0].replace(",", ""))

    date_str = date.today().strftime("%Y-%m-%d")

    if count > df["Cumulative total"].max() and date_str > df["Date"].max():
        new = pd.DataFrame(
            {
                "Country": "Laos",
                "Date": [date_str],
                "Cumulative total": count,
                "Source URL": url,
                "Source label": "Government of Laos",
                "Units": "people tested",
            }
        )

        df = pd.concat([new, df], sort=False)

    df.to_csv("automated_sheets/Laos.csv", index=False)


if __name__ == "__main__":
    main()
