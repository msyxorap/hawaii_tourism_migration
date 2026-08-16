"""
05_build_panel.py
Merge the four processed datasets into the county-year analysis panel.

Inputs  (data/processed/):
    irs_hawaii_migration_1990_2023.csv      <- dependent variable source
    dbedt_hawaii_visitor_days_1999_2024.csv <- tourism exposure
    laus_hawaii_unemployment_1990_2025.csv  <- labor market control
    fhfa_hawaii_hpi.csv                     <- housing price control

Output:
    data/processed/panel_county_year.csv

Panel design notes
------------------
* The IRS migration data defines the panel spine: 4 counties x 1990-2022
  (year = first year of the IRS filing-year pair, so 1990 = the 1990-91
  migration year).
* Out-migration rates use the standard IRS population-at-risk denominator:
  households that could have left = non-migrant returns + all out-migrant
  returns. Rates are computed for all out-migration and for the
  different-state component (the headline DV: leaving Hawaii entirely).
* DBEDT visitor days start in 1999, so ln_visitor_days is missing for
  1990-1998. Those rows are KEPT in the panel (controls-only estimation,
  robustness) and the coverage map below shows exactly where the
  estimation sample begins.
* The IRS `era` column marks the two documented series breaks (2011-12
  methodology change, 2022-23 matching update) so 06 can add era controls.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")

COUNTIES = ["Hawaii County", "Honolulu County", "Kauai County", "Maui County"]

# ---------------------------------------------------------------- load ----
irs = pd.read_csv(PROCESSED / "irs_hawaii_migration_1990_2023.csv")
dbedt = pd.read_csv(PROCESSED / "dbedt_hawaii_visitor_days_1999_2024.csv")
laus = pd.read_csv(PROCESSED / "laus_hawaii_unemployment_1990_2025.csv")
fhfa = pd.read_csv(PROCESSED / "fhfa_hawaii_hpi.csv")

# ------------------------------------------------- IRS: rates, wide form ----
# One row per county-year with in- and out-flows side by side.
keep = [
    "county", "year", "era",
    "total_us_foreign_returns", "total_us_foreign_exemptions",
    "same_state_returns", "diff_state_returns", "diff_state_exemptions",
    "foreign_returns", "nonmigrant_returns", "nonmigrant_exemptions",
]
wide = (
    irs[keep + ["direction"]]
    .pivot(index=["county", "year", "era"], columns="direction")
)
wide.columns = [f"{col}_{direction}" for col, direction in wide.columns]
wide = wide.reset_index()

# Population at risk of out-migrating: stayers + all leavers (returns = households).
wide["households_at_risk"] = (
    wide["nonmigrant_returns_out"] + wide["total_us_foreign_returns_out"]
)

# Out-migration rates (per household at risk).
wide["out_rate_all"] = wide["total_us_foreign_returns_out"] / wide["households_at_risk"]
wide["out_rate_diff_state"] = wide["diff_state_returns_out"] / wide["households_at_risk"]
wide["out_rate_same_state"] = wide["same_state_returns_out"] / wide["households_at_risk"]

# In-migration and net (for descriptives / robustness).
wide["in_rate_all"] = wide["total_us_foreign_returns_in"] / wide["households_at_risk"]
wide["net_migration_returns"] = (
    wide["total_us_foreign_returns_in"] - wide["total_us_foreign_returns_out"]
)
wide["net_rate"] = wide["net_migration_returns"] / wide["households_at_risk"]

# Household size proxy (exemptions per return, non-migrants).
wide["avg_household_size"] = (
    wide["nonmigrant_exemptions_out"] / wide["nonmigrant_returns_out"]
)

irs_panel = wide[[
    "county", "year", "era", "households_at_risk",
    "out_rate_all", "out_rate_diff_state", "out_rate_same_state",
    "in_rate_all", "net_rate", "net_migration_returns", "avg_household_size",
]]

# ---------------------------------------------------- DBEDT: counties only ----
tourism = (
    dbedt.loc[dbedt["county"].isin(COUNTIES), ["county", "year", "visitor_days"]]
    .copy()
)
tourism["ln_visitor_days"] = np.log(tourism["visitor_days"])

# ------------------------------------------------------------- LAUS ----
labor = laus[["county", "year", "unemployment_rate", "labor_force"]].copy()
labor["ln_labor_force"] = np.log(labor["labor_force"])

# ------------------------------------------------------------- FHFA ----
housing = fhfa[["county", "year", "hpi_1990_base"]].copy()
housing["ln_hpi"] = np.log(housing["hpi_1990_base"])

# ------------------------------------------------------------- merge ----
panel = (
    irs_panel
    .merge(tourism, on=["county", "year"], how="left")
    .merge(labor, on=["county", "year"], how="left")
    .merge(housing, on=["county", "year"], how="left")
    .sort_values(["county", "year"])
    .reset_index(drop=True)
)

# Convenience: log of the DV for log-log specifications (rates are all > 0).
panel["ln_out_rate_diff_state"] = np.log(panel["out_rate_diff_state"])

out_path = PROCESSED / "panel_county_year.csv"
panel.to_csv(out_path, index=False)

# ------------------------------------------------------ coverage map ----
print("#lemme seee it")
print(f"\nPanel written to {out_path}")
print(f"Rows: {len(panel)}  (counties: {panel.county.nunique()}, "
      f"years: {panel.year.min()}-{panel.year.max()})")

print("\nCOVERAGE MAP  (M = migration DV, T = tourism, U = unemployment, H = house prices)")
print("A row of 'MTUH' means the observation is fully usable in the main specification.\n")
header = "year  " + "  ".join(f"{c.split()[0]:>8}" for c in COUNTIES)
print(header)
print("-" * len(header))
for year in range(int(panel.year.min()), int(panel.year.max()) + 1):
    cells = []
    for county in COUNTIES:
        row = panel[(panel.county == county) & (panel.year == year)]
        if row.empty:
            cells.append("....")
            continue
        r = row.iloc[0]
        cell = (
            ("M" if pd.notna(r.out_rate_diff_state) else ".")
            + ("T" if pd.notna(r.ln_visitor_days) else ".")
            + ("U" if pd.notna(r.unemployment_rate) else ".")
            + ("H" if pd.notna(r.ln_hpi) else ".")
        )
        cells.append(cell)
    print(f"{year}  " + "  ".join(f"{c:>8}" for c in cells))

full = panel.dropna(subset=["out_rate_diff_state", "ln_visitor_days",
                            "unemployment_rate", "ln_hpi"])
print(f"\nFull-information observations (MTUH): {len(full)}  "
      f"({full.year.min()}-{full.year.max()})")
print(f"Controls-only observations (M.UH, pre-DBEDT): {len(panel) - len(full)}")

print("\nDV summary by county (out_rate_diff_state, full sample):")
print(panel.groupby("county")["out_rate_diff_state"]
      .agg(["mean", "min", "max"]).round(4).to_string())

print("\nEra composition (for series-break controls in 06):")
print(panel.groupby("era")["year"].agg(["min", "max", "count"]).to_string())
