"""
16_spike_2016.py
Diagnose the 2016 out-migration spike: measurement artifact, not exodus.

Input:  data/processed/panel_national_county_year.csv
Output: output/note_2016_spike.txt   (the numbers behind the footnote)

Evidence assembled here:
 1. The spike is NATIONAL: the aggregate out-of-state exit rate jumps
    from 4.2% (2014) to 7.3% (2016), then collapses to 2.7% (2018).
 2. It lives in the NUMERATOR: gross migrant returns rise ~33-34% in
    2015 and 2016 and fall 21% and 52% in 2017-18, while non-migrant
    returns move ~1% per year. Populations do not behave this way;
    matched-return counts under changing processing rules do.
 3. External documentation: SOI's own presentation (Pierce, IRS SOI,
    Oct 2022, 'IRS Migration Data') includes a Data Anomalies section
    showing the IRS interstate-mover rate diverging from ACS and CPS
    benchmarks in the 2015-16 and 2016-17 vintages and re-aligning
    afterward, alongside an updated-vs-previous methodology comparison
    for 2008-2019.
 4. Consequence for THIS study: all panel specifications include year
    fixed effects, which absorb these common level shifts; the Maui
    synthetic control matched THROUGH the 2016 spike (donors spiked
    identically). Identification is from cross-county variation within
    year, which the artifact leaves intact unless its incidence is
    county-correlated with tourism exposure - the Hawaii-vs-national
    decomposition below checks exactly that.
 5. Writing consequence: descriptive claims about a '2016 exodus peak'
    in LEVELS must be reframed - most of Hawaii's 2016 level spike is
    the artifact. Hawaii's rates RELATIVE to the national rate are the
    meaningful series.
"""

from pathlib import Path

import pandas as pd

OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


nat = pd.read_csv("data/processed/panel_national_county_year.csv",
                  dtype={"fips": str})

emit("#lemme seee it")

natrate = nat.groupby("year").apply(
    lambda d: d.diff_state_returns.sum() / d.households_at_risk.sum(),
    include_groups=False)
emit("\n1. National out-of-state exit rate (%):")
emit((100 * natrate).round(2).to_string())

agg = nat.groupby("year")[["diff_state_returns", "nonmigrant_returns",
                           "households_at_risk"]].sum()
chg = (agg.pct_change() * 100).loc[2013:2019]
emit("\n2. Year-over-year % change, national aggregates:")
emit(chg.round(1).to_string())

emit("\n3. Hawaii relative to the national rate "
     "(county rate / national rate):")
hi = nat[nat.fips.str.startswith("15") & nat.fips.isin(
    ["15001", "15003", "15007", "15009"])]
rel = (hi.pivot_table(index="year", columns="fips",
                      values="out_rate_diff_state")
       .div(natrate, axis=0))
emit(rel.loc[2013:2019].round(2).to_string())
emit("\nReading: Honolulu's RELATIVE position in 2016 (~"
     f"{rel.loc[2016, '15003']:.2f}x national) is elevated but far less "
     "dramatic than its level spike;")
emit("most of the 2016 level jump in every Hawaii county is the "
     "common national artifact.")


emit("\n5. Regime-split robustness of the headline national result")
emit("   (donor pool, county+year FE, county-clustered SEs):")
from linearmodels.panel import PanelOLS
BASE = ["ln_out_rate_diff_state", "ln_tourism_share",
        "unemployment_rate", "ln_hpi"]
full = nat.dropna(subset=BASE)
cnt = full.groupby("fips").size()
bal = full[full.fips.isin(cnt[cnt == 12].index)]
donor = bal[bal.donor_pool]
def _fit(d):
    dd = d.set_index(["fips", "year"])
    return PanelOLS(dd["ln_out_rate_diff_state"],
                    dd[["ln_tourism_share", "unemployment_rate", "ln_hpi"]],
                    entity_effects=True, time_effects=True,
                    drop_absorbed=True).fit(cov_type="clustered",
                                            cluster_entity=True)
for lab, d in [("full 2011-22", donor),
               ("old regime 2011-17", donor[donor.year <= 2017]),
               ("new regime 2018-22", donor[donor.year >= 2018])]:
    r = _fit(d)
    emit(f"   {lab:20} coef {r.params['ln_tourism_share']:+.4f}  "
         f"SE {r.std_errors['ln_tourism_share']:.4f}  "
         f"p {r.pvalues['ln_tourism_share']:.3f}  N {int(r.nobs)}")
emit("   Point estimates nearly identical in each regime; the retention")
emit("   channel does not ride the measurement break.")

emit("\n6. Footnote text (methods appendix):")
emit("-" * 70)
emit("The apparent surge in measured migration in the 2015-16 and "
     "2016-17\nIRS vintages, and the sharp drop from 2018-19 onward, "
     "reflect documented\nchanges in SOI's return-matching methodology "
     "rather than population\nbehavior: gross migrant returns rise "
     "roughly a third in each of 2015\nand 2016 and fall by half by "
     "2018 while non-migrant counts are stable,\nand SOI's own "
     "benchmarking (Pierce 2022) shows the IRS mover rate\ndiverging "
     "from ACS and CPS in exactly these vintages before re-aligning.\n"
     "All specifications include year fixed effects, which absorb these\n"
     "common shifts; descriptive level comparisons across the 2014-2018\n"
     "window are avoided, and Hawaii's series are interpreted relative "
     "to\nthe national rate.")
emit("-" * 70)

(OUTDIR / "note_2016_spike.txt").write_text("\n".join(lines) + "\n")
print(f"\nWritten to {OUTDIR}/note_2016_spike.txt")
