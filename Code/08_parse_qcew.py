"""
08_parse_qcew.py
Parse QCEW annual county files into a national tourism-exposure panel.

Input:
    data/raw/qcew/{year}.zip  for 1990-2025, each containing the two BLS
    annual by-industry CSVs extracted unmodified from
    {year}_annual_by_industry.zip (bls.gov/cew/downloadable-data-files.htm):
        "... 10 ... Total, all industries.csv"
        "... 72 ... Accommodation and food services.csv"

Output:
    data/processed/qcew_tourism_share_1990_2025.csv
    One row per county-year:
        fips, county_name, state_fips, year,
        emp_total   (annual avg employment, all industries, all ownerships),
        emp_afs     (annual avg employment, NAICS 72, private),
        wage_afs    (avg weekly wage, NAICS 72, private),
        tourism_share = emp_afs / emp_total

Selection rules (verified against the raw files):
    * County rows: 5-digit area_fips not ending in 000 (state/US totals),
      not ending in 999 (Unknown/Undefined), not starting with C (CSAs).
    * Totals:   own_code 0 (all ownerships), agglvl_code 70.
    * NAICS 72: own_code 5 (private),        agglvl_code 74.
    * Disclosure: counties suppressed by BLS are absent in older vintages
      and flagged disclosure_code == 'N' (with zeros) in newer ones; both
      end up as missing emp_afs here, never as a fake zero.
    * NAICS series for 1990-2000 are BLS reconstructions from SIC records
      (documented on the BLS download page); note this in the methods text.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/qcew")
OUT = Path("data/processed/qcew_tourism_share_1990_2025.csv")

YEARS = range(1990, 2026)

USECOLS = ["area_fips", "own_code", "industry_code", "agglvl_code",
           "disclosure_code", "area_title", "annual_avg_emplvl",
           "annual_avg_wkly_wage"]


def county_rows(df: pd.DataFrame) -> pd.DataFrame:
    f = df["area_fips"].astype(str)
    return df[(f.str.len() == 5) & f.str.isdigit()
              & ~f.str.endswith("000") & ~f.str.endswith("999")].copy()


def read_member(z: zipfile.ZipFile, key: str) -> pd.DataFrame:
    name = [n for n in z.namelist() if key in n and n.endswith(".csv")]
    assert len(name) == 1, f"expected one '{key}' file, found {name}"
    with z.open(name[0]) as fh:
        return pd.read_csv(io.TextIOWrapper(fh, "utf-8"),
                           usecols=USECOLS, dtype={"area_fips": str})


frames = []
for year in YEARS:
    zpath = RAW / f"{year}.zip"
    if not zpath.exists():
        print(f"  !! {year}.zip missing — skipped")
        continue
    with zipfile.ZipFile(zpath) as z:
        tot = read_member(z, "Total, all industries")
        afs = read_member(z, "Accommodation")

    tot = county_rows(tot)
    tot = tot[(tot.own_code == 0) & (tot.agglvl_code == 70)]
    tot = tot.rename(columns={"annual_avg_emplvl": "emp_total"})

    afs = county_rows(afs)
    afs = afs[(afs.own_code == 5) & (afs.agglvl_code == 74)]
    # BLS suppression: 'N' rows carry zeros that are NOT real zeros.
    afs.loc[afs.disclosure_code.astype(str).eq("N"),
            ["annual_avg_emplvl", "annual_avg_wkly_wage"]] = pd.NA
    afs = afs.rename(columns={"annual_avg_emplvl": "emp_afs",
                              "annual_avg_wkly_wage": "wage_afs"})

    merged = tot[["area_fips", "area_title", "emp_total"]].merge(
        afs[["area_fips", "emp_afs", "wage_afs"]],
        on="area_fips", how="left")
    merged["year"] = year
    frames.append(merged)

panel = pd.concat(frames, ignore_index=True)
panel = panel.rename(columns={"area_fips": "fips", "area_title": "county_name"})
panel["state_fips"] = panel["fips"].str[:2]
panel["emp_afs"] = pd.to_numeric(panel["emp_afs"], errors="coerce")
panel["tourism_share"] = panel["emp_afs"] / panel["emp_total"]
# Zero total employment (a handful of tiny counties) -> undefined share.
panel.loc[panel.emp_total == 0, "tourism_share"] = pd.NA

panel = panel[["fips", "county_name", "state_fips", "year",
               "emp_total", "emp_afs", "wage_afs", "tourism_share"]]
panel.to_csv(OUT, index=False)

# ------------------------------------------------------------- report ----
print("#lemme seee it")
print(f"\nWritten to {OUT}")
print(f"Rows: {len(panel):,}  counties: {panel.fips.nunique():,}  "
      f"years: {panel.year.min()}-{panel.year.max()}")
print(f"emp_afs coverage: {panel.emp_afs.notna().mean():.1%} of county-years "
      f"(rest BLS-suppressed or no NAICS-72 establishments)")

hi = panel[panel.fips.str.startswith("15")]
print("\nHawaii tourism_share (sanity check):")
print(hi[hi.year.isin([1990, 2005, 2019, 2020, 2024])]
      .pivot(index="year", columns="county_name", values="tourism_share")
      .round(3).to_string())

y19 = panel[(panel.year == 2019) & (panel.emp_total > 5000)]
print("\nTop 10 tourism-share counties, 2019 (emp_total > 5,000):")
print(y19.nlargest(10, "tourism_share")
      [["county_name", "tourism_share", "emp_total"]]
      .round(3).to_string(index=False))
