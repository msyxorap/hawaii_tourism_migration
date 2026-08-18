"""
12_synthetic_maui.py
Synthetic control: Maui's out-of-state migration around the COVID shock.

Input:  data/processed/panel_national_county_year.csv   (09)
Outputs:
    output/table_synth.txt     numeric results (weights, gaps, placebo p)
    output/synth_paths.csv     actual/synthetic paths for 13_figures.py

Method
------
Classic Abadie-style synthetic control, matching the full pre-treatment
outcome path (Ferman-Pinto style; with all pre-period outcomes as
predictors, additional covariates are redundant - Kaul et al. 2022 -
noted for the methods text).

* Outcome: out_rate_diff_state (level).
* Treated unit: Maui (15009). Secondary runs: Honolulu, Hawaii County.
* Treatment year: 2020 (first COVID migration year under the first-year
  convention). Pre-period: 2011-2019. Post: 2020-2022.
* Donor pool: the 133 MAINLAND donor-pool counties with complete
  outcomes 2011-2022 (all Hawaii counties excluded from every donor set
  to avoid within-state spillovers).
* Weights: nonnegative, sum to one; solved by NNLS with a sum-to-one
  augmentation row, then normalized (deterministic, no tuning).
* Inference: in-space placebos. Every donor county is treated as if it
  were Maui; the test statistic is the post/pre RMSPE ratio, and the
  p-value is Maui's rank in that distribution (Abadie, Diamond,
  Hainmueller 2010).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)

PRE = list(range(2011, 2020))
POST = list(range(2020, 2023))
YEARS = PRE + POST
OUTCOME = "out_rate_diff_state"

nat = pd.read_csv("data/processed/panel_national_county_year.csv",
                  dtype={"fips": str})
wide = (nat.pivot_table(index="fips", columns="year", values=OUTCOME)
        .reindex(columns=YEARS))
names = nat.drop_duplicates("fips").set_index("fips")["county_name"]

donor_ids = sorted(set(nat[nat.donor_pool].fips) - {"15009"})
donor_ids = [f for f in donor_ids if not f.startswith("15")
             and f in wide.index and wide.loc[f, YEARS].notna().all()]
D = wide.loc[donor_ids, YEARS]          # donors x years
print("#lemme seee it")
print(f"\nDonor pool: {len(donor_ids)} mainland counties, "
      f"pre {PRE[0]}-{PRE[-1]}, post {POST[0]}-{POST[-1]}")


def synth(treated_path: pd.Series, donors: pd.DataFrame):
    """Weights on donors matching the treated pre-period path."""
    A = donors[PRE].to_numpy().T                       # pre-years x donors
    b = treated_path[PRE].to_numpy()
    lam = 1000.0                                       # sum-to-one row
    A_aug = np.vstack([A, lam * np.ones(A.shape[1])])
    b_aug = np.append(b, lam)
    w, _ = nnls(A_aug, b_aug)
    w = w / w.sum()
    path = donors[YEARS].to_numpy().T @ w
    return w, pd.Series(path, index=YEARS)


def rmspe(actual, synthetic, years):
    d = actual[years].to_numpy() - synthetic[years].to_numpy()
    return float(np.sqrt(np.mean(d ** 2)))


def run_unit(fips, donors_df, donor_list):
    actual = wide.loc[fips]
    w, s = synth(actual, donors_df)
    pre_r = rmspe(actual, s, PRE)
    post_r = rmspe(actual, s, POST)
    return w, s, pre_r, post_r, (post_r / pre_r if pre_r > 0 else np.nan)


# ------------------------------------------------------------ Maui ----
w, s_maui, pre_r, post_r, ratio_maui = run_unit("15009", D, donor_ids)
actual_maui = wide.loc["15009"]

print("\nMAUI: actual vs synthetic out-of-state out-migration rate")
tab = pd.DataFrame({"actual": actual_maui, "synthetic": s_maui,
                    "gap": actual_maui - s_maui}).round(4)
print(tab.to_string())
print(f"\npre-RMSPE {pre_r:.5f}   post-RMSPE {post_r:.5f}   "
      f"ratio {ratio_maui:.2f}")

top_w = pd.Series(w, index=donor_ids).sort_values(ascending=False).head(8)
print("\nSynthetic Maui is built from:")
for f, wt in top_w.items():
    if wt > 0.005:
        print(f"  {wt:5.1%}  {names.get(f, f)}")

# ------------------------------------------------- placebo inference ----
ratios = {}
for f in donor_ids:
    others = D.drop(index=f)
    _, _, pr, po, rat = run_unit(f, others, list(others.index))
    ratios[f] = rat
placebo = pd.Series(ratios).dropna()
p_maui = (1 + (placebo >= ratio_maui).sum()) / (1 + len(placebo))
print(f"\nPlacebo inference: Maui ratio {ratio_maui:.2f} vs "
      f"{len(placebo)} placebo counties")
print(f"  placebo ratio median {placebo.median():.2f}, "
      f"p90 {placebo.quantile(.9):.2f}")
print(f"  p-value, all placebos (rank of Maui): {p_maui:.3f}")

# Standard ADH refinements.
# (i) Ratios explode when a placebo's pre-RMSPE ~ 0 (near-exact
#     interpolation with 132 donors); restrict to placebos whose pre-fit
#     is within 5x of Maui's so ratios are comparable.
pre_fits = {}
for f in donor_ids:
    others = D.drop(index=f)
    _, _, pr, _, _ = run_unit(f, others, list(others.index))
    pre_fits[f] = pr
pre_fits = pd.Series(pre_fits)
ok = placebo[pre_fits[placebo.index] <= 5 * pre_r]
p_flt = (1 + (ok >= ratio_maui).sum()) / (1 + len(ok))
print(f"  p-value, comparable-fit placebos (n={len(ok)}): {p_flt:.3f}")

# (ii) Gap-based statistic: rank of Maui's 2020 gap among placebo 2020
#      gaps (sign-specific; avoids ratio denominators entirely).
gaps20 = {}
for f in donor_ids:
    others = D.drop(index=f)
    _, s_p, _, _, _ = run_unit(f, others, list(others.index))
    gaps20[f] = wide.loc[f, 2020] - s_p[2020]
gaps20 = pd.Series(gaps20)
g_maui = actual_maui[2020] - s_maui[2020]
p_gap = (1 + (gaps20 >= g_maui).sum()) / (1 + len(gaps20))
print(f"  2020 gap: Maui {g_maui:+.4f} vs placebo median "
      f"{gaps20.median():+.4f}; p (rank) {p_gap:.3f}")

# ------------------------------------------- secondary Hawaii units ----
print("\nSecondary units (same donor pool):")
paths = {"year": YEARS, "maui_actual": actual_maui[YEARS].to_numpy(),
         "maui_synth": s_maui[YEARS].to_numpy()}
for fips, lab in [("15003", "Honolulu"), ("15001", "Hawaii County")]:
    if fips in wide.index and wide.loc[fips, YEARS].notna().all():
        _, s, pr, po, rat = run_unit(fips, D, donor_ids)
        p = (1 + (placebo >= rat).sum()) / (1 + len(placebo))
        print(f"  {lab:14} ratio {rat:5.2f}   placebo p {p:.3f}")
        paths[f"{lab.split()[0].lower()}_actual"] = wide.loc[fips, YEARS].to_numpy()
        paths[f"{lab.split()[0].lower()}_synth"] = s[YEARS].to_numpy()
    else:
        print(f"  {lab:14} incomplete outcome path — skipped")

pd.DataFrame(paths).to_csv(OUTDIR / "synth_paths.csv", index=False)

with open(OUTDIR / "table_synth.txt", "w") as fh:
    fh.write(tab.to_string())
    fh.write(f"\n\nMaui ratio {ratio_maui:.2f}, placebo p {p_maui:.3f}\n")
    fh.write("Weights:\n")
    fh.write("\n".join(f"  {wt:5.1%}  {names.get(f, f)}"
                       for f, wt in top_w.items() if wt > 0.005))
print(f"\nWritten to {OUTDIR}/table_synth.txt and synth_paths.csv")
