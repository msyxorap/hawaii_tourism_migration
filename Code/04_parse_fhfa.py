#Imports
import zipfile
import pandas as pd
from pathlib import Path
import io

# FHFA annual house price index by county (All-Transactions, developmental)
# one xlsx inside the zip, columns:
# State | County | FIPS code | Year | Annual Change (%) | HPI |
# HPI with 1990 base | HPI with 2000 base
# just keep the HI rows. the 1990-base column is the one the panel will use
# since the panel starts in 1990.

#where the data lives
RAW_ZIP = "data/raw/fhfa.zip"
OUT_CSV = "data/processed/fhfa_hawaii_hpi.csv"

#fhfa county names to my county names (fips codes come along for the ride)
county_names = {
    "15001": "Hawaii County",
    "15003": "Honolulu County",
    "15005": "Maui County",   #Kalawao, basically never has an index
    "15007": "Kauai County",
    "15009": "Maui County",
}

#calls
with zipfile.ZipFile(RAW_ZIP) as z:
    xlsx_name = ""
    for name in z.namelist():
        if name.lower().endswith(".xlsx"):
            xlsx_name = name
    fhfa_df = pd.read_excel(io.BytesIO(z.read(xlsx_name)), skiprows=6)

fhfa_df.columns = ["state", "county_name", "fips", "year", "annual_change_pct",
                   "hpi", "hpi_1990_base", "hpi_2000_base"]

#keep hawaii only
fhfa_df = fhfa_df[fhfa_df["state"].astype(str).str.strip() == "HI"].copy()
fhfa_df["county"] = fhfa_df["fips"].astype(str).str.strip().map(county_names)

#numbers come in as text sometimes with '.' for missing
for column in ["annual_change_pct", "hpi", "hpi_1990_base", "hpi_2000_base"]:
    fhfa_df[column] = pd.to_numeric(fhfa_df[column], errors="coerce")

#Kalawao never has its own index so the maui merge is just keeping real maui,
#groupby mean would be wrong for an index anyway, so drop empty kalawao rows
fhfa_df = fhfa_df.dropna(subset=["hpi"])
fhfa_df = fhfa_df[["county", "year", "annual_change_pct", "hpi", "hpi_1990_base", "hpi_2000_base"]]

fhfa_df = fhfa_df.sort_values(["county", "year"])
Path("data/processed").mkdir(parents=True, exist_ok=True)
fhfa_df.to_csv(OUT_CSV, index=False)

#lemme seee it
print("wrote", OUT_CSV, "with", len(fhfa_df), "rows, years", int(fhfa_df["year"].min()), "-", int(fhfa_df["year"].max()))
print(fhfa_df.pivot_table(index="year", columns="county", values="hpi_1990_base").round(1).tail(40).to_string())
