"""
13_figures.py
Dissertation figures and descriptive statistics (styled).
Outputs: output/figures/fig1..fig4.png, output/table_descriptives.txt
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

FIGDIR = Path("output/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

hi = pd.read_csv("data/processed/panel_county_year.csv")
nat = pd.read_csv("data/processed/panel_national_county_year.csv",
                  dtype={"fips": str, "state_fips": str})
paths = pd.read_csv("output/synth_paths.csv")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.titlesize": 12.5, "axes.titleweight": "semibold",
    "axes.titlepad": 12, "axes.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#d0d4da", "grid.linewidth": 0.6, "grid.alpha": 0.6,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "legend.frameon": False,
})
BLUE, RED, GRAY, BAND = "#2563c9", "#c2402f", "#6e7681", "#f1e2c0"
HI_COLORS = {"Hawaii County": "#2d7f5e", "Honolulu County": "#2563c9",
             "Kauai County": "#8250df", "Maui County": "#c2402f"}


def endlabel(ax, x, y, text, color, dx=0.15, fs=9.5):
    ax.annotate(text, (x, y), xytext=(x + dx, y), color=color,
                fontsize=fs, fontweight="semibold", va="center")


# ---------------------------------------------- fig 1: synthetic Maui ----
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7.4, 6.2), sharex=True,
    gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.12})
for ax in (ax1, ax2):
    ax.axvspan(2019.5, 2022.4, color=BAND, alpha=0.45, zorder=0)
ax1.plot(paths.year, 100 * paths.maui_actual, color=RED, lw=2.4,
         marker="o", ms=5, zorder=3)
ax1.plot(paths.year, 100 * paths.maui_synth, color=BLUE, lw=2.2, ls="--",
         marker="s", ms=4.5, zorder=3)
endlabel(ax1, 2022, 100 * paths.maui_actual.iloc[-1] + 0.12,
         "Maui", RED)
endlabel(ax1, 2022, 100 * paths.maui_synth.iloc[-1] - 0.18,
         "Synthetic\nMaui", BLUE)
ax1.text(2021.0, 4.62, "COVID era", color="#8a6d1d", fontsize=9,
         ha="center", style="italic")
ax1.annotate("2020: +0.33 pp above\ncounterfactual, then reverts",
             xy=(2020, 4.06), xytext=(2017.1, 4.42), fontsize=9,
             color="#333333",
             arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
ax1.set_ylabel("Out-of-state out-migration rate (%)")
ax1.set_xlim(2010.6, 2023.4)
ax1.set_title("Maui and its synthetic counterpart, 2011–2022")
gap = 100 * (paths.maui_actual - paths.maui_synth)
ax2.bar(paths.year, gap, width=0.62, zorder=3,
        color=[RED if g > 0 else BLUE for g in gap], alpha=0.85)
ax2.axhline(0, color="#444444", lw=0.8)
ax2.set_ylabel("Gap (pp)")
ax2.set_xlabel("Migration year (first year of IRS filing pair)")
fig.text(0.13, -0.015,
         "Synthetic Maui: Jo Daviess IL 32%, Leelanau MI 20%, "
         "Dukes MA (Martha\u2019s Vineyard) 18%, Pitkin CO (Aspen) 15%, "
         "Inyo CA 10%. Pre-period RMSPE 0.0004.",
         fontsize=8.3, color=GRAY)
fig.savefig(FIGDIR / "fig1_synthetic_maui.png")
plt.close(fig)

# ------------------------------------------- fig 2: Hawaii series ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 6.4), sharex=True,
                               gridspec_kw={"hspace": 0.16})
events = [(2001, 2002, "9/11"), (2008, 2010, "GFC"),
          (2020, 2021.6, "COVID")]
for ax in (ax1, ax2):
    for a, b, _ in events:
        ax.axvspan(a, b, color=BAND, alpha=0.35, zorder=0)

def spread(targets, min_gap):
    """Nudge label y-positions apart until no pair overlaps."""
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    ys = [targets[i] for i in order]
    for _ in range(50):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_gap:
                ys[i - 1] -= (min_gap - (ys[i] - ys[i - 1])) / 2
                ys[i] += (min_gap - (ys[i] - ys[i - 1])) / 2
                moved = True
        if not moved:
            break
    out = [0.0] * len(targets)
    for pos, i in enumerate(order):
        out[i] = ys[pos]
    return out

ends1, ends2, cols, labs = [], [], [], []
for c, col in HI_COLORS.items():
    d = hi[hi.county == c].sort_values("year")
    v = d.dropna(subset=["visitor_days"])
    base = v[v.year == 1999].visitor_days.iloc[0]
    s1 = 100 * v.visitor_days / base
    ax1.plot(v.year, s1, color=col, lw=2.2, zorder=3,
             solid_capstyle="round")
    ax1.plot(v.year.iloc[-1], s1.iloc[-1], "o", color=col, ms=5, zorder=4)
    s2 = 100 * d.out_rate_diff_state
    ax2.plot(d.year, s2, color=col, lw=2.2, zorder=3,
             solid_capstyle="round")
    ax2.plot(d.year.iloc[-1], s2.iloc[-1], "o", color=col, ms=5, zorder=4)
    ends1.append(s1.iloc[-1]); ends2.append(s2.iloc[-1])
    cols.append(col); labs.append(c.replace(" County", ""))
for ys, ax, gap in [(spread(ends1, 9), ax1, 9),
                    (spread(ends2, 0.32), ax2, 0.32)]:
    for y, col, lab in zip(ys, cols, labs):
        ax.annotate(lab, (2022.4, y), color=col, fontsize=9.5,
                    fontweight="semibold", va="center")
for a, _, lab in events:
    ax1.text((a), 174, lab, color="#8a6d1d", fontsize=8.5,
             style="italic", ha="left")
ax1.axhline(100, color=GRAY, lw=0.7, ls=":")
ax1.text(1990, 103, "1999 level", color=GRAY, fontsize=8)
ax1.annotate("\u221256% in 2020", xy=(2020, 46), xytext=(2013.6, 60),
             fontsize=9, color="#333333",
             arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
ax1.set_ylabel("Visitor days (1999 = 100)")
ax1.set_ylim(38, 180)
ax1.set_xlim(1989, 2027)
ax1.set_title("Tourism volume and out-of-state migration in Hawai\u02bbi")
ax1.text(0.01, 0.94, "A. Visitor volume", transform=ax1.transAxes,
         fontsize=10, fontweight="semibold", color="#333333")
ax2.text(0.01, 0.94, "B. Out-of-state exit rate", transform=ax2.transAxes,
         fontsize=10, fontweight="semibold", color="#333333")
ax2.set_ylabel("Out-of-state exit rate (%)")
ax2.set_xlabel("Year")
fig.savefig(FIGDIR / "fig2_hawaii_series.png")
plt.close(fig)

# --------------------------------------- fig 3: exit geography ----
BASE = ["ln_out_rate_diff_state", "ln_tourism_share",
        "unemployment_rate", "ln_hpi"]
full = nat.dropna(subset=BASE)
cnt = full.groupby("fips").size()
bal_ids = cnt[cnt == full.year.nunique()].index
donor = full[(full.donor_pool & full.fips.isin(bal_ids))
             | (full.fips.str.startswith("15")
                & full.fips.isin(cnt[cnt >= 11].index))]
comp = (donor.assign(ds=lambda d: d.diff_state_returns
                     / (d.diff_state_returns + d.same_state_returns))
        .groupby(["fips", "county_name"])["ds"].mean().reset_index())
mainland = comp[~comp.fips.str.startswith("15")]
hi_rows = comp[comp.fips.str.startswith("15")].sort_values("ds")

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.hist(mainland.ds, bins=24, color=BLUE, alpha=0.5, zorder=2,
        edgecolor="white", linewidth=0.6)
ax.set_ylim(0, 14.5)
m = mainland.ds.mean()
ax.axvline(m, color=GRAY, ls="--", lw=1.1, zorder=3)
ax.annotate(f"mainland mean {m:.0%}", xy=(m, 12.4), xytext=(m - 0.24, 12.9),
            fontsize=9, color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
heights = {0: 10.4, 1: 8.4, 2: 12.2}
for k, (_, r) in enumerate(hi_rows.iterrows()):
    nm = r.county_name.split(" County")[0].replace(" + Kalawao", "")
    ax.axvline(r.ds, color=RED, lw=2.2, zorder=4)
    ax.annotate(f"{nm}  {r.ds:.0%}", xy=(r.ds, heights[k]),
                xytext=(r.ds - 0.135, heights[k] + 1.1), fontsize=9.5,
                color=RED, fontweight="semibold",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Share of within-US out-moves that cross a state line, "
              "2011–2022 mean")
ax.set_ylabel("Counties")
ax.set_title("The geography of exit: where tourism-county leavers go")
ax.text(0.115, 13.9, f"{len(mainland)} mainland tourism-dependent counties",
        fontsize=9, color=BLUE)
fig.savefig(FIGDIR / "fig3_exit_geography.png")
plt.close(fig)

# ------------------------------------ fig 4: national binned scatter ----
sub = full[full.fips.isin(bal_ids)].copy()
for v in ["ln_out_rate_diff_state", "ln_tourism_share"]:
    r = sub[v].copy()
    for _ in range(6):
        r = r - r.groupby(sub.fips).transform("mean")
        r = r - r.groupby(sub.year).transform("mean")
    sub[v + "_r"] = r
q = pd.qcut(sub.ln_tourism_share_r, 25, duplicates="drop")
b = sub.groupby(q, observed=True).agg(
    x=("ln_tourism_share_r", "mean"),
    y=("ln_out_rate_diff_state_r", "mean"))
slope = np.polyfit(sub.ln_tourism_share_r, sub.ln_out_rate_diff_state_r, 1)[0]
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.scatter(b.x, b.y, color=BLUE, s=42, zorder=3, edgecolor="white",
           linewidth=0.8)
xs = np.linspace(b.x.min(), b.x.max(), 50)
ax.plot(xs, slope * xs, color=RED, lw=1.8, zorder=2)
ax.annotate(f"within-county, within-year slope = {slope:.2f}",
            xy=(0.05, 0.02), fontsize=10, color=RED,
            fontweight="semibold")
ax.axhline(0, color=GRAY, lw=0.6)
ax.axvline(0, color=GRAY, lw=0.6)
ax.set_xlabel("ln tourism employment share (residualized)")
ax.set_ylabel("ln out-of-state exit rate (residualized)")
ax.set_title(f"Tourism jobs and retention: {sub.fips.nunique():,} counties, "
             "25 bins")
fig.savefig(FIGDIR / "fig4_binned_scatter.png")
plt.close(fig)

# ------------------------------------------------- descriptives ----
def desc(df, cols, labels):
    t = df[cols].describe().T[["count", "mean", "std", "min", "max"]]
    t.index = labels
    t["count"] = t["count"].astype(int)
    return t.round(4)


hi_d = hi.dropna(subset=["ln_visitor_days"]).copy()
hi_d["visitor_days_m"] = hi_d.visitor_days / 1e6
hi_d["unemployment_pct"] = 100 * hi_d.unemployment_rate
hi_t = desc(hi_d,
            ["out_rate_diff_state", "out_rate_same_state", "in_rate_all",
             "visitor_days_m", "unemployment_pct", "hpi_1990_base"],
            ["Out-of-state exit rate", "Same-state exit rate",
             "In-migration rate", "Visitor days (millions)",
             "Unemployment rate (%)", "House price index (1990=100)"])
est = nat.dropna(subset=BASE)
nat_t = desc(est,
             ["out_rate_diff_state", "out_rate_same_state",
              "tourism_share", "unemployment_rate", "hpi_1990"],
             ["Out-of-state exit rate", "Same-state exit rate",
              "Tourism employment share", "Unemployment rate (%)",
              "House price index (1990=100)"])

with open("output/table_descriptives.txt", "w") as fh:
    fh.write("PANEL A. Hawaii county-year panel (estimation sample, "
             "1999-2022, N=%d)\n\n" % len(hi_d))
    fh.write(hi_t.to_string())
    fh.write("\n\nPANEL B. National county-year panel (full-information "
             "sample, 2011-2022, N=%d, counties=%d)\n\n"
             % (len(est), est.fips.nunique()))
    fh.write(nat_t.to_string())

print("#lemme seee it")
print("\nFigures written to output/figures/:")
for f in sorted(FIGDIR.glob("*.png")):
    print("  ", f.name)
print("\nDescriptives written to output/table_descriptives.txt")
print("\n" + open("output/table_descriptives.txt").read())
