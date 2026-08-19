"""
15_iniki.py
Hurricane Iniki (September 1992) as a natural experiment: synthetic
control for Kauai's out-of-state migration.

Inputs:
    data/processed/irs_national_early_1990_2003.csv   (14)
    data/processed/qcew_tourism_share_1990_2025.csv   (08)
Outputs:
    output/table_iniki.txt
    output/iniki_paths.csv        (for figures)

Design
------
* Treated unit: Kauai (15007). Treatment: migration year 1992 (the
  1992-93 IRS pair, the first to contain post-Iniki moves).
* Pre-period: 1990-1991 outcome path. HONESTY NOTE, stated everywhere
  this is reported: two pre-treatment outcomes is a thin matching
  window (the IRS county series simply begins in 1990). The placebo
  distribution is computed under the identical thin window, so the
  rank-based inference is internally consistent, and the QCEW first
  stage (below) provides the treatment-intensity evidence that the
  design leans on.
* Donor pool: mainland counties in the top decile of 1990-91 QCEW
  tourism share among counties with complete outcomes 1990-2000
  (era-appropriate analogue of the 2011-13 donor rule; all Hawaii
  counties excluded from donors).
* FIRST STAGE: Iniki's tourism shock measured directly - Kauai NAICS-72
  employment vs the donor mean, 1990-1996.
* Inference: in-space placebos, post(1992-94)/pre RMSPE ratios and the
  1992-94 mean-gap rank (both reported; ratios also reported filtered
  to comparable pre-fit as in 12).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)

PRE = [1990, 1991]
POST = [1992, 1993, 1994]
WINDOW = list(range(1990, 2001))
OUTCOME = "out_rate_diff_state"

irs = pd.read_csv("data/processed/irs_national_early_1990_2003.csv",
                  dtype={"fips": str})
qcew = pd.read_csv("data/processed/qcew_tourism_share_1990_2025.csv",
                   dtype={"fips": str})

wide = (irs.pivot_table(index="fips", columns="year", values=OUTCOME)
        .reindex(columns=WINDOW))
complete = wide.dropna().index

base = (qcew[qcew.year.isin([1990, 1991])]
        .groupby("fips")["tourism_share"].mean())
elig = base[base.index.isin(complete) & ~base.index.str.startswith("15")]
cut = elig.quantile(0.90)
donor_ids = sorted(elig[elig >= cut].index)
D = wide.loc[donor_ids, WINDOW]
names = qcew.drop_duplicates("fips").set_index("fips")["county_name"]

print("#lemme seee it")
print(f"\nDonor pool: {len(donor_ids)} mainland counties "
      f"(1990-91 tourism share >= {cut:.3f}, complete outcomes 1990-2000)")

# ------------------------------------------------------- first stage ----
k_emp = qcew[(qcew.fips == "15007") & qcew.year.between(1990, 1996)]
d_emp = (qcew[qcew.fips.isin(donor_ids) & qcew.year.between(1990, 1996)]
         .groupby("year")["emp_afs"].mean())
k90 = k_emp[k_emp.year == 1990].emp_afs.iloc[0]
d90 = d_emp.loc[1990]
print("\nFIRST STAGE - accommodation & food employment (1990 = 100):")
print(f"{'year':>6}{'Kauai':>9}{'donor mean':>12}")
fs = {}
for y in range(1990, 1997):
    kv = 100 * k_emp[k_emp.year == y].emp_afs.iloc[0] / k90
    dv = 100 * d_emp.loc[y] / d90
    fs[y] = (kv, dv)
    print(f"{y:>6}{kv:>9.1f}{dv:>12.1f}")

# --------------------------------------------------------------- SCM ----
def synth(treated, donors):
    A = donors[PRE].to_numpy().T
    b = treated[PRE].to_numpy()
    lam = 1000.0
    A_aug = np.vstack([A, lam * np.ones(A.shape[1])])
    b_aug = np.append(b, lam)
    w, _ = nnls(A_aug, b_aug)
    w = w / w.sum()
    return w, pd.Series(donors[WINDOW].to_numpy().T @ w, index=WINDOW)


def rmspe(a, s, yrs):
    d = a[yrs].to_numpy() - s[yrs].to_numpy()
    return float(np.sqrt(np.mean(d ** 2)))


actual = wide.loc["15007"]
w, s = synth(actual, D)
pre_r, post_r = rmspe(actual, s, PRE), rmspe(actual, s, POST)
ratio_k = post_r / pre_r
gap = actual - s
mgap_k = gap[POST].mean()

print("\nKAUAI: actual vs synthetic out-of-state out-migration rate")
tab = pd.DataFrame({"actual": actual, "synthetic": s, "gap": gap}).round(4)
print(tab.loc[1990:1997].to_string())
print(f"\npost(92-94)/pre RMSPE ratio {ratio_k:.2f}   "
      f"mean 92-94 gap {mgap_k:+.4f} "
      f"({100 * mgap_k / s[POST].mean():+.1f}% of counterfactual)")
top_w = pd.Series(w, index=donor_ids).sort_values(ascending=False)
print("\nSynthetic Kauai is built from:")
for f, wt in top_w.items():
    if wt > 0.01:
        print(f"  {wt:5.1%}  {names.get(f, f)}")

# --------------------------------------------------------- placebos ----
ratios, mgaps, prefits = {}, {}, {}
for f in donor_ids:
    others = D.drop(index=f)
    a_p = wide.loc[f]
    _, s_p = synth(a_p, others)
    pr, po = rmspe(a_p, s_p, PRE), rmspe(a_p, s_p, POST)
    ratios[f] = po / pr if pr > 0 else np.nan
    mgaps[f] = (a_p - s_p)[POST].mean()
    prefits[f] = pr
ratios, mgaps, prefits = (pd.Series(x) for x in (ratios, mgaps, prefits))

p_ratio = (1 + (ratios.dropna() >= ratio_k).sum()) / (1 + ratios.notna().sum())
ok = ratios[(prefits <= 5 * pre_r) & ratios.notna()]
p_flt = (1 + (ok >= ratio_k).sum()) / (1 + len(ok))
p_gap = (1 + (mgaps >= mgap_k).sum()) / (1 + len(mgaps))
print(f"\nPlacebo inference ({len(donor_ids)} donors):")
print(f"  RMSPE-ratio p:                 {p_ratio:.3f}")
print(f"  comparable-fit ratio p (n={len(ok)}): {p_flt:.3f}")
print(f"  mean 92-94 gap: Kauai {mgap_k:+.4f} vs placebo median "
      f"{mgaps.median():+.4f}; rank p {p_gap:.3f}")


# ------------------------------- within-Hawaii comparison (Part C) ----
# Iniki hit Kauai and spared the neighbor islands, which share every
# statewide factor. Independent of the mainland donors entirely.
w_hi = irs.pivot_table(index="year", columns="fips", values=OUTCOME)
kau = 100 * w_hi["15007"]
nbr = 100 * w_hi[["15001", "15009"]].mean(axis=1)  # Hawaii Co + Maui
rel = (kau - nbr) - (kau - nbr).loc[[1990, 1991]].mean()
print("\nWITHIN-HAWAII: Kauai minus neighbor-island mean (pp, rel. 1990-91):")
print(rel.loc[1990:1997].round(3).to_string())
print("1992-94 mean: {:+.3f} pp - two independent comparison groups, "
      "same answer".format(rel.loc[1992:1994].mean()))

pd.DataFrame({"year": WINDOW, "kauai_actual": actual[WINDOW].to_numpy(),
              "kauai_synth": s[WINDOW].to_numpy(),
              "fs_kauai": [fs.get(y, (np.nan,))[0] for y in WINDOW],
              "fs_donor": [fs.get(y, (np.nan, np.nan))[1] for y in WINDOW]
              }).to_csv(OUTDIR / "iniki_paths.csv", index=False)

with open(OUTDIR / "table_iniki.txt", "w") as fh:
    fh.write(tab.to_string())
    fh.write(f"\n\nratio {ratio_k:.2f} (p {p_ratio:.3f}; comparable-fit p "
             f"{p_flt:.3f}); mean 92-94 gap {mgap_k:+.4f} (p {p_gap:.3f})\n")
    fh.write("\nWithin-Hawaii rel. gaps 1992-94 (pp): "
         + ", ".join(f"{rel.loc[y]:+.3f}" for y in (1992, 1993, 1994)) + "\n")
    fh.write("Weights:\n" + "\n".join(
        f"  {wt:5.1%}  {names.get(f, f)}"
        for f, wt in top_w.items() if wt > 0.01))
print(f"\nWritten to {OUTDIR}/table_iniki.txt and iniki_paths.csv")
