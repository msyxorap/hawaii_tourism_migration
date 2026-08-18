"""
07_robustness.py
Robustness and inference program for the baseline tourism-migration result.

Input:  data/processed/panel_county_year.csv
Output: output/table_robustness.txt (and printed summary)

What this does, and why (committee-proofing the baseline)
---------------------------------------------------------
A. INFERENCE. With G = 4 clusters, cluster-robust SEs are not reliable.
   Three responses, all reported:
     1. Wild cluster bootstrap p-values (Cameron-Gelbach-Miller 2008),
        using 6-point Webb weights (recommended when G is very small).
        With 4 clusters the weight space is fully enumerable (6^4 = 1296
        draws), so the bootstrap is exact, not simulated.
     2. Driscoll-Kraay SEs (robust to cross-sectional dependence and
        serial correlation; asymptotics in T = 24, not G = 4).
     3. HC1 as the optimistic benchmark.
B. TIMING. IRS year t covers moves between filing year t and t+1, so
   contemporaneous visitor days is the natural regressor; the lag
   specification checks whether exposure takes a year to bite.
C. COVID. 2020-22 contain a 70% collapse and rebound in visitor days.
   The pre-COVID sample (1999-2019) checks the result is not one shock.
D. SIZE. Honolulu is ~2/3 of the state. WLS by households at risk checks
   the unweighted result is not driven by the small counties.
E. TRENDS. County-specific linear trends guard against slow-moving
   confounders (identification then comes only off deviations from trend).
F. MECHANISM. Same model on all-destination out-migration and on
   IN-migration. Displacement story: out rises, in does not rise as much
   (or falls). Pure churn story: both rise together.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

PANEL = Path("data/processed/panel_county_year.csv")
OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)

panel = pd.read_csv(PANEL).sort_values(["county", "year"])
panel["ln_visitor_days_lag"] = panel.groupby("county")["ln_visitor_days"].shift(1)
panel["ln_out_rate_all"] = np.log(panel["out_rate_all"])
panel["ln_in_rate_all"] = np.log(panel["in_rate_all"])
panel["year_f"] = panel["year"].astype("category")
panel["trend"] = panel["year"] - panel["year"].min()

BASE = ["ln_out_rate_diff_state", "ln_visitor_days",
        "unemployment_rate", "ln_hpi"]
sample = panel.dropna(subset=BASE).copy()

CONTROLS = "unemployment_rate + ln_hpi + C(county) + C(year_f)"


# ---------------------------------------------------- wild cluster boot ----
def wild_cluster_boot_p(formula, data, param, cluster_col="county",
                        weights=None):
    """Exact Webb-weight wild cluster bootstrap p-value (null imposed)."""
    webb = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                     np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    clusters = data[cluster_col].unique()
    G = len(clusters)

    # NOTE: bootstrap t-stats are studentized with HC1, not the cluster
    # covariance. With G = 4 the cluster covariance is rank-deficient and
    # its diagonal is numerically unstable (results differed across
    # machines/library versions, including nan SEs). The cluster
    # correction enters through the CLUSTER-LEVEL weight draws below;
    # HC1 studentization keeps the statistic well-defined and makes the
    # procedure exactly reproducible.
    def tstat(df):
        if weights is None:
            m = smf.ols(formula, data=df).fit(cov_type="HC1")
        else:
            m = smf.wls(formula, data=df, weights=df[weights]).fit(
                cov_type="HC1")
        return m.params[param] / m.bse[param], m

    t_obs, m_full = tstat(data)

    # Restricted model (null: coefficient on param = 0) for residuals.
    rhs = formula.split("~")[1]
    terms = [t.strip() for t in rhs.split("+") if t.strip() != param]
    f_restr = formula.split("~")[0] + "~ " + " + ".join(terms)
    if weights is None:
        m_r = smf.ols(f_restr, data=data).fit()
    else:
        m_r = smf.wls(f_restr, data=data, weights=data[weights]).fit()
    fitted_r, resid_r = m_r.fittedvalues, m_r.resid

    yvar = formula.split("~")[0].strip()
    t_boot = []
    for draw in product(webb, repeat=G):
        w = data[cluster_col].map(dict(zip(clusters, draw))).to_numpy()
        db = data.copy()
        db[yvar] = fitted_r + resid_r * w
        t_b, _ = tstat(db)
        t_boot.append(t_b)
    t_boot = np.array(t_boot)
    return float(np.mean(np.abs(t_boot) >= np.abs(t_obs))), m_full


# ------------------------------------------------------------- spec grid ----
specs = [
    ("(1) Baseline",            "ln_out_rate_diff_state", "ln_visitor_days",
     f"ln_out_rate_diff_state ~ ln_visitor_days + {CONTROLS}", sample, None),
    ("(2) Lagged tourism",      "ln_out_rate_diff_state", "ln_visitor_days_lag",
     f"ln_out_rate_diff_state ~ ln_visitor_days_lag + {CONTROLS}",
     panel.dropna(subset=BASE + ["ln_visitor_days_lag"]), None),
    ("(3) Pre-COVID (99-19)",   "ln_out_rate_diff_state", "ln_visitor_days",
     f"ln_out_rate_diff_state ~ ln_visitor_days + {CONTROLS}",
     sample[sample.year <= 2019], None),
    ("(4) WLS (hh weights)",    "ln_out_rate_diff_state", "ln_visitor_days",
     f"ln_out_rate_diff_state ~ ln_visitor_days + {CONTROLS}",
     sample, "households_at_risk"),
    ("(5) County trends",       "ln_out_rate_diff_state", "ln_visitor_days",
     f"ln_out_rate_diff_state ~ ln_visitor_days + {CONTROLS}"
     " + C(county):trend", sample, None),
    ("(6) DV: all out-mig",     "ln_out_rate_all", "ln_visitor_days",
     f"ln_out_rate_all ~ ln_visitor_days + {CONTROLS}", sample, None),
    ("(7) DV: IN-migration",    "ln_in_rate_all", "ln_visitor_days",
     f"ln_in_rate_all ~ ln_visitor_days + {CONTROLS}", sample, None),
]

print("#lemme seee it")
print("\nROBUSTNESS GRID — coefficient on tourism exposure")
print("(wild-bootstrap p uses exact Webb weights, 1296 draws, null imposed)\n")
hdr = (f"{'spec':24}{'coef':>9}{'HC1 SE':>9}{'HC1 p':>8}"
       f"{'boot p':>8}{'N':>5}")
print(hdr)
print("-" * len(hdr))

rows = []
for name, dv, param, formula, data, wcol in specs:
    boot_p, m_cl = wild_cluster_boot_p(formula, data, param, weights=wcol)
    if wcol is None:
        m_hc1 = smf.ols(formula, data=data).fit(cov_type="HC1")
    else:
        m_hc1 = smf.wls(formula, data=data, weights=data[wcol]).fit(
            cov_type="HC1")
    row = dict(spec=name, coef=m_hc1.params[param], hc1_se=m_hc1.bse[param],
               hc1_p=m_hc1.pvalues[param], boot_p=boot_p, n=int(m_hc1.nobs))
    rows.append(row)
    print(f"{name:24}{row['coef']:>9.4f}{row['hc1_se']:>9.4f}"
          f"{row['hc1_p']:>8.3f}{row['boot_p']:>8.3f}{row['n']:>5d}")

# ------------------------------------------- Driscoll-Kraay, main spec ----
pdat = sample.set_index(["county", "year"])
dk = PanelOLS.from_formula(
    "ln_out_rate_diff_state ~ ln_visitor_days + unemployment_rate + ln_hpi"
    " + EntityEffects + TimeEffects", data=pdat
).fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
b = dk.params["ln_visitor_days"]
print(f"\nDriscoll-Kraay (main spec): coef {b:.4f}, "
      f"SE {dk.std_errors['ln_visitor_days']:.4f}, "
      f"p {dk.pvalues['ln_visitor_days']:.3f}")

# ---------------------------------------------------------------- write ----
tab = pd.DataFrame(rows).round(4)
with open(OUTDIR / "table_robustness.txt", "w") as fh:
    fh.write(tab.to_string(index=False))
    fh.write(f"\n\nDriscoll-Kraay (main): coef {b:.4f}, "
             f"SE {dk.std_errors['ln_visitor_days']:.4f}, "
             f"p {dk.pvalues['ln_visitor_days']:.3f}\n")
print(f"\nWritten to {OUTDIR}/table_robustness.txt")