import datetime
from selenium import webdriver
import pandas as pd
import dateparser


SOURCE_URL = "https://covid19.saglik.gov.tr/TR-66935/genel-koronavirus-tablosu.html"


def main():

    with webdriver.Firefox() as driver:
        driver.get(SOURCE_URL)

        table = driver.find_element_by_class_name("table-striped")
        df = pd.read_html(table.get_attribute("outerHTML"), thousands=".")[0]
        df = df[["Tarih", "Bugünkü Test Sayısı"]].dropna()

        df = df.rename(
            columns={
                "Tarih": "Date",
                "Toplam Test Sayısı": "Cumulative total",
                "Bugünkü Test Sayısı": "Daily change in cumulative total",
            }
        )

        df["Date"] = (
            df["Date"]
            .str.replace("MAYIS", "mayıs")
            .str.replace("KASIM", "kasım")
            .str.replace("ARALIK", "aralık")
            .apply(dateparser.parse, languages=["tr"])
        )
        df.loc[:, "Country"] = "Turkey"
        df.loc[:, "Units"] = "tests performed"
        df.loc[:, "Source URL"] = SOURCE_URL
        df.loc[:, "Source label"] = "Turkish Ministry of Health"
        df.loc[:, "Notes"] = pd.NA

        df.to_csv("automated_sheets/Turkey.csv", index=False)


if __name__ == "__main__":
    main()
