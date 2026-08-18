"""
09_build_panel_national.py
Build the national county-year analysis panel, 2011-2022.

Inputs:
    data/raw/irs_national/{Y1}-{Y2}/countyoutflow{y1y2}.csv   (2011-12 .. 2022-23)
    data/raw/irs_national/{Y1}-{Y2}/countyinflow{y1y2}.csv
    data/processed/qcew_tourism_share_1990_2025.csv           (built by 08)
    data/raw/laus.zip      (laucntyYY.xlsx, national county LAUS)
    data/raw/fhfa.zip      (hpi_at_county.xlsx, national county HPI)

Output:
    data/processed/panel_national_county_year.csv

Design notes
------------
* Year convention matches the Hawaii panel: year = first year of the IRS
  filing pair, so year 2011 = the 2011-12 migration year. Modern IRS CSVs
  cover 2011-2022 under this convention.
* IRS summary rows are identified by destination pseudo-state codes
  (96 = total US+foreign, 97 = US / same-state / different-state splits,
  98 = foreign) with the label distinguishing the 97-group rows, and the
  non-migrant row by origin == destination with a 'Non-migrant' label.
  -1 values are IRS suppression -> missing, never fake zeros.
* Denominator (households at risk) = non-migrant returns + total out
  returns, as in the Hawaii panel (05).
* tourism_share_base = county mean tourism share over 2011-2013; the
  donor_pool flag marks the top decile of that baseline among counties
  with a full DV. Threshold choice is revisited in 10; the continuous
  baseline is stored so nothing is locked in.
* LAUS covers 1990-2024 and FHFA 1975-2025 nationally; both are parsed
  in full here (cheap) so the 1990s backfill can reuse this script.
"""

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
IRS = RAW / "irs_national"
PROCESSED = Path("data/processed")
OUT = PROCESSED / "panel_national_county_year.csv"

YEARS = range(2011, 2023)  # first-year labels for 2011-12 .. 2022-23

# ------------------------------------------------------------ IRS ----

def yy(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def read_irs(year: int, direction: str) -> pd.DataFrame:
    """One row per focal county with non-migrant / total / same / diff /
    foreign returns (n1) and exemptions (n2)."""
    path = IRS / f"{year}-{year + 1}" / f"county{direction}flow{yy(year)}.csv"
    if not path.exists():
        print(f"  !! {path.name} missing — {direction}-flow for {year} "
              f"left as NaN")
        return pd.DataFrame(columns=["fips", "year"])
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.lower()
    # focal county = origin (y1) in outflow files, destination (y2) in inflow
    foc, oth = ("y1", "y2") if direction == "out" else ("y2", "y1")
    df["fips"] = (df[f"{foc}_statefips"].astype(int).astype(str).str.zfill(2)
                  + df[f"{foc}_countyfips"].astype(int).astype(str).str.zfill(3))
    name = f"{oth}_countyname"
    lab = df[name].astype(str).str.lower()
    ostate = df[f"{oth}_statefips"].astype(int)

    def grab(mask, tag):
        g = df.loc[mask, ["fips", "n1", "n2"]].copy()
        g[["n1", "n2"]] = g[["n1", "n2"]].where(g[["n1", "n2"]] >= 0)  # -1 = suppressed
        return g.rename(columns={"n1": f"{tag}_returns", "n2": f"{tag}_exemptions"})

    parts = [
        grab(lab.str.contains("non-migrant"), "nonmigrant"),
        grab(ostate.eq(96), "total"),
        grab(ostate.eq(97) & lab.str.contains("same state"), "same_state"),
        grab(ostate.eq(97) & lab.str.contains("different state"), "diff_state"),
        grab(ostate.eq(98), "foreign"),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="fips", how="outer")
    out["year"] = year
    return out


irs_out, irs_in = [], []
for y in YEARS:
    irs_out.append(read_irs(y, "out"))
    irs_in.append(read_irs(y, "in"))
o = pd.concat(irs_out, ignore_index=True)
i = pd.concat(irs_in, ignore_index=True)

mig = o.merge(i[["fips", "year", "total_returns", "total_exemptions"]],
              on=["fips", "year"], how="left", suffixes=("_out", "_in"))
mig["households_at_risk"] = mig["nonmigrant_returns"] + mig["total_returns_out"]
for tag in ["total_returns_out", "diff_state_returns", "same_state_returns"]:
    short = tag.replace("_returns", "").replace("total_out", "all")
    mig[f"out_rate_{'all' if 'total' in tag else short}"] = (
        mig[tag] / mig["households_at_risk"])
mig["in_rate_all"] = mig["total_returns_in"] / mig["households_at_risk"]
mig["net_rate"] = mig["in_rate_all"] - mig["out_rate_all"]

# ------------------------------------------------------------ QCEW ----
qcew = pd.read_csv(PROCESSED / "qcew_tourism_share_1990_2025.csv",
                   dtype={"fips": str})

# ------------------------------------------------------------ LAUS ----
laus_frames = []
with zipfile.ZipFile(RAW / "laus.zip") as z:
    for member in z.namelist():
        m = re.search(r"laucnty(\d{2})\.(xlsx?|xls)$", member)
        if not m:
            continue
        d = pd.read_excel(io.BytesIO(z.read(member)), header=None, skiprows=2,
                          names=["laus_code", "st", "cty", "name", "year",
                                 "labor_force", "employed", "unemployed",
                                 "unemployment_rate"])
        d = d.dropna(subset=["st", "cty", "year"])
        d["fips"] = (d.st.astype(int).astype(str).str.zfill(2)
                     + d.cty.astype(int).astype(str).str.zfill(3))
        laus_frames.append(d[["fips", "year", "labor_force",
                              "unemployment_rate"]])
laus = pd.concat(laus_frames, ignore_index=True)
laus["year"] = laus["year"].astype(int)
for c in ["labor_force", "unemployment_rate"]:
    laus[c] = pd.to_numeric(laus[c], errors="coerce")

# ------------------------------------------------------------ FHFA ----
with zipfile.ZipFile(RAW / "fhfa.zip") as z:
    member = [n for n in z.namelist() if n.endswith(".xlsx")][0]
    fh = pd.read_excel(io.BytesIO(z.read(member)), skiprows=6, header=None,
                       names=["state", "county", "fips", "year",
                              "annual_change", "hpi", "hpi_1990", "hpi_2000"])
fh = fh.dropna(subset=["fips", "year"])
fh["fips"] = fh["fips"].astype(str).str.zfill(5)
fh["year"] = fh["year"].astype(int)
fh["hpi_1990"] = pd.to_numeric(fh["hpi_1990"], errors="coerce")
fhfa = fh[["fips", "year", "hpi_1990"]]

# ----------------------------------------------------------- merge ----
panel = (mig
         .merge(qcew[["fips", "year", "county_name", "state_fips",
                      "emp_total", "tourism_share"]],
                on=["fips", "year"], how="left")
         .merge(laus, on=["fips", "year"], how="left")
         .merge(fhfa, on=["fips", "year"], how="left"))

# Logged variables (rates > 0 required for logs; zeros stay missing).
for c, ln in [("out_rate_diff_state", "ln_out_rate_diff_state"),
              ("out_rate_all", "ln_out_rate_all"),
              ("tourism_share", "ln_tourism_share"),
              ("hpi_1990", "ln_hpi"),
              ("labor_force", "ln_labor_force")]:
    panel[ln] = np.log(panel[c].where(panel[c] > 0))

# Baseline tourism dependence and donor pool flag.
base = (panel[panel.year.between(2011, 2013)]
        .groupby("fips")["tourism_share"].mean()
        .rename("tourism_share_base"))
panel = panel.merge(base, on="fips", how="left")

full_dv = panel.dropna(subset=["ln_out_rate_diff_state", "ln_tourism_share",
                               "unemployment_rate", "ln_hpi"])
counties_full = full_dv.groupby("fips").size()
balanced = counties_full[counties_full == len(list(YEARS))].index
cutoff = (base[base.index.isin(balanced)].quantile(0.90))
panel["donor_pool"] = (panel["tourism_share_base"] >= cutoff) & \
                      panel["fips"].isin(balanced)

panel.to_csv(OUT, index=False)

# ---------------------------------------------------------- report ----
print("#lemme seee it")
print(f"\nWritten to {OUT}")
print(f"Rows: {len(panel):,}   counties: {panel.fips.nunique():,}   "
      f"years: {panel.year.min()}-{panel.year.max()}")

est = panel.dropna(subset=["ln_out_rate_diff_state", "ln_tourism_share",
                           "unemployment_rate", "ln_hpi"])
print(f"\nFull-information estimation sample: {len(est):,} county-years, "
      f"{est.fips.nunique():,} counties")
print(f"Balanced full-DV counties (all {len(list(YEARS))} years): "
      f"{len(balanced):,}")
print(f"Donor pool (top decile baseline tourism share, balanced): "
      f"{panel[panel.donor_pool].fips.nunique():,} counties "
      f"(baseline share >= {cutoff:.3f})")

hi = panel[panel.fips.str.startswith("15")]
print("\nHawaii cross-check vs Hawaii panel (out_rate_diff_state):")
print(hi.pivot_table(index="year", columns="fips",
                     values="out_rate_diff_state").round(4).to_string())

print("\nTop 10 donor-pool counties by baseline tourism share:")
dp = (panel[panel.donor_pool]
      .drop_duplicates("fips").nlargest(10, "tourism_share_base"))
print(dp[["fips", "county_name", "tourism_share_base"]]
      .round(3).to_string(index=False))
