"""
11_hawaii_contrast.py
The geography of displacement: where do tourism-county leavers go?

Inputs:
    data/processed/panel_national_county_year.csv   (09)
    data/processed/panel_county_year.csv            (05, Hawaii)
Output:
    output/table_contrast.txt

Three parts:

A. THE DESCRIPTIVE FACT. Among tourism-dependent (donor-pool) counties,
   what share of each county's within-US out-moves cross a state line?
   Mainland resort counties have neighboring counties and in-state metros
   to absorb the displaced; Hawaii does not. If isolation shapes exit
   geography, Hawaii should sit at the extreme top of this distribution.
   This is a levels fact - no identification assumptions at all.

B. PRESSURE -> COMPOSITION. Within donor-pool counties: when housing
   pressure rises (ln HPI, within county+year), which exit margin moves?
   Prediction: mainland displacement flows into SAME-state moves; the
   Hawaii interaction should tilt the response toward DIFFERENT-state.
   (HPI is an equilibrium object, so this is channel evidence, not causal
   identification - stated as such.)

C. HAWAII VOLUME COMPOSITION. Consolidates the Hawaii-panel result that
   visitor-day volume raises different-state exits, lowers same-state
   exits, and leaves totals flat (the pattern parts A-B rationalize).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)
report = []


def emit(line=""):
    print(line)
    report.append(line)


nat = pd.read_csv("data/processed/panel_national_county_year.csv",
                  dtype={"fips": str, "state_fips": str}).sort_values(["fips", "year"])
hi_panel = pd.read_csv("data/processed/panel_county_year.csv")

BASE = ["ln_out_rate_diff_state", "ln_tourism_share",
        "unemployment_rate", "ln_hpi"]
full = nat.dropna(subset=BASE)
cnt = full.groupby("fips").size()
balanced = full[full.fips.isin(cnt[cnt == full.year.nunique()].index)]

# Contrast sample: mainland donor pool PLUS every Hawaii county with at
# least 11 full-information years. Stated rule, stated reason: Hawaii is
# the case of interest; Honolulu misses the top-decile cutoff by 0.0004
# and Hawaii County by a single missing year, and excluding them would
# leave the "Hawaii" interaction identified off Maui alone. Kauai has no
# QCEW tourism share (BLS suppression) and cannot enter.
hi_ok = cnt[(cnt >= full.year.nunique() - 1)
            & cnt.index.str.startswith("15")].index
donor = full[(full.donor_pool & full.fips.isin(balanced.fips.unique()))
             | full.fips.isin(hi_ok)].copy()

emit("#lemme seee it")

# ---------------------------------------------------- A: the fact ----
donor["diff_share_of_us_moves"] = (
    donor.diff_state_returns / (donor.diff_state_returns
                                + donor.same_state_returns))
comp = (donor.groupby(["fips", "county_name"])["diff_share_of_us_moves"]
        .mean().reset_index().sort_values("diff_share_of_us_moves",
                                          ascending=False))
mainland = comp[~comp.fips.str.startswith("15")]

emit("\nA. SHARE OF WITHIN-US OUT-MOVES THAT CROSS A STATE LINE")
emit("   (contrast sample = mainland donor pool + Hawaii counties with")
emit("   >=11 full years; 2011-2022 mean)")
emit("   Note: raw cross-state shares also reflect state geography -")
emit("   counties near borders of small states (Clark NV, Teton WY) rank")
emit("   high mechanically. Hawaii's position is the conservative reading:")
emit("   it ranks near the top DESPITE having no state border at all.\n")
emit("Top 12 of 134:")
emit(comp.head(12).round(3).to_string(index=False))
emit(f"\nHawaii counties:            "
     f"{comp[comp.fips.str.startswith('15')].diff_share_of_us_moves.min():.3f}"
     f" - {comp[comp.fips.str.startswith('15')].diff_share_of_us_moves.max():.3f}")
emit(f"Mainland donor mean:        {mainland.diff_share_of_us_moves.mean():.3f}")
emit(f"Mainland donor p10 - p90:   "
     f"{mainland.diff_share_of_us_moves.quantile(.10):.3f} - "
     f"{mainland.diff_share_of_us_moves.quantile(.90):.3f}")
emit(f"Hawaii ranks (of {len(comp)}): "
     f"{sorted(comp.reset_index(drop=True).index[comp.reset_index(drop=True).fips.str.startswith('15')] + 1)}")

# ------------------------------------------ B: pressure->composition ----
donor["ln_out_rate_same_state"] = np.log(
    donor.out_rate_same_state.where(donor.out_rate_same_state > 0))
donor["hawaii"] = donor.fips.str.startswith("15").astype(float)
donor["ln_hpi_x_hi"] = donor.ln_hpi * donor.hawaii


def fe(d, dv, xv):
    dd = d.dropna(subset=[dv] + xv).set_index(["fips", "year"])
    return PanelOLS(dd[dv], dd[xv], entity_effects=True, time_effects=True,
                    drop_absorbed=True).fit(cov_type="clustered",
                                            cluster_entity=True)


emit("\n\nB. HOUSING PRESSURE AND EXIT GEOGRAPHY (donor pool, county+year FE)")
emit("   coefficient on ln HPI (and Hawaii differential) by exit margin\n")
emit(f"{'DV':28}{'ln_hpi':>9}{'p':>7}{'x Hawaii':>10}{'p':>7}")
for dv, lab in [("ln_out_rate_same_state", "same-state exits"),
                ("ln_out_rate_diff_state", "different-state exits")]:
    r = fe(donor, dv, ["ln_hpi", "ln_hpi_x_hi", "ln_tourism_share",
                       "unemployment_rate"])
    emit(f"{lab:28}{r.params['ln_hpi']:>9.3f}{r.pvalues['ln_hpi']:>7.3f}"
         f"{r.params['ln_hpi_x_hi']:>10.3f}{r.pvalues['ln_hpi_x_hi']:>7.3f}")

# ------------------------------------------------ C: Hawaii volume ----
hp = hi_panel.dropna(subset=["ln_visitor_days"]).copy()
hp["ln_same"] = np.log(hp.out_rate_same_state)
hp["ln_all"] = np.log(hp.out_rate_all)
hp["year_f"] = hp.year.astype("category")
ctrl = "unemployment_rate + ln_hpi + C(county) + C(year_f)"

emit("\n\nC. HAWAII: VISITOR-DAY VOLUME AND EXIT MARGINS (1999-2022, "
     "county+year FE,\n   county-clustered SEs; 4 clusters - see ch. "
     "inference discussion)\n")
emit(f"{'DV':28}{'elasticity':>11}{'p':>7}")
for dv, lab in [("ln_out_rate_diff_state", "different-state exits"),
                ("ln_same", "same-state exits"),
                ("ln_all", "all exits")]:
    m = smf.ols(f"{dv} ~ ln_visitor_days + {ctrl}", data=hp).fit(
        cov_type="cluster", cov_kwds={"groups": hp["county"]})
    emit(f"{lab:28}{m.params['ln_visitor_days']:>11.3f}"
         f"{m.pvalues['ln_visitor_days']:>7.3f}")

(OUTDIR / "table_contrast.txt").write_text("\n".join(report) + "\n")
print(f"\nWritten to {OUTDIR}/table_contrast.txt")
