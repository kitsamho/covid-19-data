import pandas as pd


def main():
    pcr = pd.read_csv(
        "https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto49/Positividad_Diaria_Media_T.csv"
    )
    ag = pd.read_csv(
        "https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto49/Positividad_Diaria_Media_Ag_T.csv"
    )

    pcr = pcr[["Fecha", "pcr"]]
    ag = ag[["Fecha", "Ag"]]

    df = pd.merge(ag, pcr, on="Fecha", how="outer").fillna(0).sort_values("Fecha")
    df["Daily change in cumulative total"] = df["Ag"] + df["pcr"]
    df = df.rename(columns={"Fecha": "Date"})
    df = df.drop(["Ag", "pcr"], axis=1)

    df["Country"] = "Chile"
    df[
        "Source URL"
    ] = "https://github.com/MinCiencia/Datos-COVID19/tree/master/output/producto49"
    df["Source label"] = "Chile Ministry of Health"
    df["Units"] = "tests performed"
    df["Notes"] = pd.NA

    df.to_csv("automated_sheets/Chile.csv", index=False)


if __name__ == "__main__":
    main()
