"""
10_regressions_national.py
National panel regressions: tourism exposure and out-migration, 2011-2022.

Input:  data/processed/panel_national_county_year.csv   (built by 09)
Output: output/table_national.txt

Design
------
All models: two-way fixed effects (county + year), estimated by within
transformation (linearmodels PanelOLS), SEs clustered by county unless
noted. Exposure is ln(tourism_share) = ln(NAICS-72 emp / total emp), so
coefficients are elasticities w.r.t. tourism *intensity* (the Hawaii
chapter's ln visitor-days elasticity is the level analogue; both are
within-county, within-year comparisons).

Specs:
 (1) Full sample          all counties with complete data (unbalanced)
 (2) Balanced sample      counties observed all 12 years
 (3) Donor pool           tourism-dependent counties (top-decile baseline)
 (4) Donor pool, lagged   exposure at t-1 (timing preferred by theory)
 (5) Hawaii interaction   donor pool: does Hawaii's slope differ?
 (6-8) Composition        DV = same-state rate / all out / in-rate on the
                          donor pool - the displacement-geography test:
                          prediction is that mainland displacement shows
                          up in SAME-state moves, Hawaii's in DIFFERENT-
                          state moves.
State-clustered SEs are reported for the headline spec as robustness.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

PANEL = Path("data/processed/panel_national_county_year.csv")
OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(PANEL, dtype={"fips": str, "state_fips": str})
df = df.sort_values(["fips", "year"])
df["ln_tourism_share_lag"] = df.groupby("fips")["ln_tourism_share"].shift(1)
df["ln_out_rate_same_state"] = np.log(
    df["out_rate_same_state"].where(df["out_rate_same_state"] > 0))
df["ln_in_rate_all"] = np.log(df["in_rate_all"].where(df["in_rate_all"] > 0))
df["hawaii"] = df["fips"].str.startswith("15").astype(float)
df["ln_ts_x_hawaii"] = df["ln_tourism_share"] * df["hawaii"]

BASE = ["ln_out_rate_diff_state", "ln_tourism_share",
        "unemployment_rate", "ln_hpi"]
full = df.dropna(subset=BASE).copy()
counts = full.groupby("fips").size()
balanced_ids = counts[counts == full.year.nunique()].index
balanced = full[full.fips.isin(balanced_ids)]
donor = balanced[balanced.donor_pool]


def fit(data, dv, xvars, cluster="entity"):
    d = data.dropna(subset=[dv] + xvars).set_index(["fips", "year"])
    m = PanelOLS(d[dv], d[xvars], entity_effects=True, time_effects=True,
                 drop_absorbed=True)
    if cluster == "entity":
        return m.fit(cov_type="clustered", cluster_entity=True)
    grp = d["state_fips"] if cluster == "state" else None
    return m.fit(cov_type="clustered", clusters=grp)


X = ["ln_tourism_share", "unemployment_rate", "ln_hpi"]
XL = ["ln_tourism_share_lag", "unemployment_rate", "ln_hpi"]
XH = ["ln_tourism_share", "ln_ts_x_hawaii", "unemployment_rate", "ln_hpi"]

specs = [
    ("(1) Full sample",        full,     "ln_out_rate_diff_state", X,  "ln_tourism_share"),
    ("(2) Balanced",           balanced, "ln_out_rate_diff_state", X,  "ln_tourism_share"),
    ("(3) Donor pool",         donor,    "ln_out_rate_diff_state", X,  "ln_tourism_share"),
    ("(4) Donor, lagged",      donor,    "ln_out_rate_diff_state", XL, "ln_tourism_share_lag"),
    ("(5) Donor + HI inter.",  donor,    "ln_out_rate_diff_state", XH, "ln_tourism_share"),
    ("(6) DV same-state",      donor,    "ln_out_rate_same_state", X,  "ln_tourism_share"),
    ("(7) DV all out",         donor,    "ln_out_rate_all",        X,  "ln_tourism_share"),
    ("(8) DV in-rate",         donor,    "ln_in_rate_all",         X,  "ln_tourism_share"),
]

print("#lemme seee it")
print("\nNATIONAL PANEL — coefficient on tourism exposure "
      "(county+year FE, county-clustered SEs)\n")
hdr = f"{'spec':24}{'coef':>9}{'SE':>9}{'p':>8}{'N':>8}{'counties':>9}"
print(hdr)
print("-" * len(hdr))

rows, keep = [], {}
for name, data, dv, xv, param in specs:
    r = fit(data, dv, xv)
    n_cty = r.entity_info["total"]
    rows.append((name, r.params[param], r.std_errors[param],
                 r.pvalues[param], int(r.nobs), int(n_cty)))
    keep[name] = r
    print(f"{name:24}{r.params[param]:>9.4f}{r.std_errors[param]:>9.4f}"
          f"{r.pvalues[param]:>8.3f}{int(r.nobs):>8,}{int(n_cty):>9,}")

# Hawaii interaction detail
r5 = keep["(5) Donor + HI inter."]
print("\nSpec (5) detail — Hawaii vs mainland donor counties:")
print(f"  mainland slope:      {r5.params['ln_tourism_share']:>8.4f} "
      f"(p {r5.pvalues['ln_tourism_share']:.3f})")
print(f"  Hawaii differential: {r5.params['ln_ts_x_hawaii']:>8.4f} "
      f"(p {r5.pvalues['ln_ts_x_hawaii']:.3f})")
print(f"  Hawaii total slope:  "
      f"{r5.params['ln_tourism_share'] + r5.params['ln_ts_x_hawaii']:>8.4f}")

# Headline with state clustering
r3s = fit(donor, "ln_out_rate_diff_state", X, cluster="state")
print(f"\nSpec (3) with STATE-clustered SEs "
      f"({donor.state_fips.nunique()} clusters): "
      f"coef {r3s.params['ln_tourism_share']:.4f}, "
      f"SE {r3s.std_errors['ln_tourism_share']:.4f}, "
      f"p {r3s.pvalues['ln_tourism_share']:.3f}")

with open(OUTDIR / "table_national.txt", "w") as fh:
    fh.write(pd.DataFrame(rows, columns=["spec", "coef", "se", "p", "N",
                                         "counties"]).round(4).to_string(index=False))
    fh.write("\n\nSpec 5 mainland slope "
             f"{r5.params['ln_tourism_share']:.4f}, HI differential "
             f"{r5.params['ln_ts_x_hawaii']:.4f}\n")
    fh.write(f"Spec 3 state-clustered p: "
             f"{r3s.pvalues['ln_tourism_share']:.3f}\n")
print(f"\nWritten to {OUTDIR}/table_national.txt")
