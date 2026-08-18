"""
06_regressions.py
Baseline panel regressions: tourism exposure and out-of-state migration.

Input:
    data/processed/panel_county_year.csv   (built by 05_build_panel.py)

Outputs:
    output/table_main.txt    <- plain-text Stargazer table (for reading here)
    output/table_main.html   <- HTML Stargazer table (for pasting into Word)

Specification design
---------------------
DV: ln(out_rate_diff_state) — log of the share of at-risk households
    leaving Hawaii entirely in that county-year.

(1) Pooled OLS:        ln_visitor_days only
(2) + controls:        unemployment_rate, ln_hpi, era dummies
(3) + county FE:       absorbs fixed island differences (Honolulu's
                       permanently higher exit rate, base-year effects)
(4) + year FE:         absorbs statewide shocks (recessions, COVID) and
                       the IRS era breaks (era is a function of year, so
                       era dummies drop out of spec 4 automatically).
                       This is the main specification: identification
                       from within-county deviations in tourism relative
                       to the statewide year effect.

Standard errors are clustered by county in all columns. CAVEAT for the
write-up: with only 4 clusters, clustered SEs are unreliable (too few
clusters); heteroskedasticity-robust (HC1) SEs are printed in a
supplementary panel for comparison, and inference should be discussed
with that limitation stated openly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer

PANEL = Path("data/processed/panel_county_year.csv")
OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- data ----
panel = pd.read_csv(PANEL)

sample = panel.dropna(
    subset=["ln_out_rate_diff_state", "ln_visitor_days",
            "unemployment_rate", "ln_hpi"]
).copy()
sample["county"] = sample["county"].astype("category")
sample["year_f"] = sample["year"].astype("category")

print("#lemme seee it")
print(f"\nEstimation sample: {len(sample)} county-years "
      f"({sample.year.min()}-{sample.year.max()}, "
      f"{sample.county.nunique()} counties)")

# ------------------------------------------------------------- models ----
f1 = "ln_out_rate_diff_state ~ ln_visitor_days"
f2 = f1 + " + unemployment_rate + ln_hpi + C(era)"
f3 = f2 + " + C(county)"
f4 = f1 + " + unemployment_rate + ln_hpi + C(county) + C(year_f)"

formulas = [f1, f2, f3, f4]
labels = ["Pooled", "Controls", "County FE", "County + Year FE"]

models_cl, models_hc1 = [], []
for f in formulas:
    ols = smf.ols(f, data=sample)
    models_cl.append(
        ols.fit(cov_type="cluster", cov_kwds={"groups": sample["county"]})
    )
    models_hc1.append(ols.fit(cov_type="HC1"))

# ------------------------------------------------------- stargazer table ----
star = Stargazer(models_cl)
star.custom_columns(labels, [1, 1, 1, 1])
star.covariate_order(["ln_visitor_days", "unemployment_rate", "ln_hpi"])
star.rename_covariates({
    "ln_visitor_days": "ln(Visitor days)",
    "unemployment_rate": "Unemployment rate",
    "ln_hpi": "ln(House price index)",
})
star.add_line("County FE", ["No", "No", "Yes", "Yes"])
star.add_line("Year FE", ["No", "No", "No", "Yes"])
star.add_line("IRS era dummies", ["No", "Yes", "Yes", "(in year FE)"])
star.title(
    "Tourism exposure and out-of-state migration, Hawaii counties 1999-2022"
)
star.add_custom_notes([
    "DV: ln(out-of-state out-migration rate). SEs clustered by county (4 clusters).",
])

txt = star.render_latex()  # rendered below as text too
(OUTDIR / "table_main.html").write_text(star.render_html())

# Plain-text summary for the terminal: coefficient of interest across specs.
print("\n" + "=" * 74)
print("COEFFICIENT ON ln(Visitor days) ACROSS SPECIFICATIONS")
print("=" * 74)
print(f"{'':22}{'coef':>10}{'cluster SE':>12}{'p':>8}{'HC1 SE':>10}{'p':>8}")
for lab, mc, mh in zip(labels, models_cl, models_hc1):
    b = mc.params["ln_visitor_days"]
    print(f"{lab:22}{b:>10.4f}{mc.bse['ln_visitor_days']:>12.4f}"
          f"{mc.pvalues['ln_visitor_days']:>8.3f}"
          f"{mh.bse['ln_visitor_days']:>10.4f}"
          f"{mh.pvalues['ln_visitor_days']:>8.3f}")

print("\nFull results, main specification (4): County + Year FE, cluster SEs")
main = models_cl[3]
core = [p for p in main.params.index
        if not p.startswith("C(year_f)") and not p.startswith("C(county)")
        and p != "Intercept"]
summ = pd.DataFrame({
    "coef": main.params[core],
    "se": main.bse[core],
    "p": main.pvalues[core],
}).round(4)
print(summ.to_string())
print(f"\nN = {int(main.nobs)}   R2 = {main.rsquared:.3f}   "
      f"Adj R2 = {main.rsquared_adj:.3f}")

# Text version of the full table for the repo.
with open(OUTDIR / "table_main.txt", "w") as fh:
    fh.write("Cluster-by-county SEs\n" + "=" * 60 + "\n")
    for lab, m in zip(labels, models_cl):
        fh.write(f"\n--- {lab} ---\n{m.summary().as_text()}\n")
print(f"\nTables written to {OUTDIR}/table_main.html and table_main.txt")

# Elasticity interpretation at the main spec.
b4 = models_cl[3].params["ln_visitor_days"]
print(f"\nInterpretation: a 10% rise in county visitor days is associated "
      f"with a {b4 * 10:.2f}% change\nin the out-of-state out-migration rate "
      f"(spec 4, within county and year).")
