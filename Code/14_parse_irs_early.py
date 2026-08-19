"""
14_parse_irs_early.py
Parse the early-format IRS county OUTFLOW files for ALL states, 1990-2003.

This is the national generalization of 01_parse_irs.py's two early-format
cleaners (txt for 1990-91/1991-92, xls for 1992-93 onward). The logic is
identical; the only changes are (i) the home state is read from each row
instead of being fixed to Hawaii's '15', and (ii) 'same state' means the
destination state equals the county's own state.

Input:  data/raw/irs_national/{Y1}-{Y2}/{Y1}to{Y2}CountyMigrationOutflow/
        (per-state files: C9091ako.txt ... / C9293Ako.xls ...)
Output: data/processed/irs_national_early_1990_2003.csv
        fips, year, total/same/diff/foreign/nonmigrant returns+exemptions,
        out_rate_all, out_rate_diff_state, out_rate_same_state

Notes:
* Outflow only - the out-migration DV needs origin totals + non-migrants,
  all of which live in the outflow files.
* -1 values are IRS suppression -> missing (never zero).
* different-state = total - same-state - foreign, as in 01 (these vintages
  have no summary rows).
* Kalawao (15005) is folded into Maui (15009), matching every other panel.
"""

import re
from pathlib import Path

import pandas as pd
import xlrd

RAW = Path("data/raw/irs_national")
OUT = Path("data/processed/irs_national_early_1990_2003.csv")

FOREIGN = {"57", "58", "59"}


def to_number(v):
    try:
        n = float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return None if n < 0 else n


def clean_code(v):
    t = str(v).strip()
    if t == "":
        return ""
    try:
        return str(int(float(t)))
    except ValueError:
        return t


def parse_txt(path, year):
    """1990-91 / 1991-92 fixed-text files, all counties in the file."""
    header = re.compile(r"^(\d{2})\s+(\d{3})\s+(.+?)\s+Total Migration.*?"
                        r"\s+(-?[\d,]+)\s+(-?[\d,]+)\s*$")
    nonmig = re.compile(r"^\s*(\d{2})\s+(\d{3})\s+County Non-Migrants"
                        r"\s+(-?[\d,]+)\s+(-?[\d,]+)\s*$")
    dest = re.compile(r"^\s+(\d{2})\s+(\d{3})\s+(.+?)\s+([A-Za-z]{2})"
                      r"\s+(-?[\d,]+)\s+[\d.]*\s*(-?[\d,]+)\s+[\d.]*\s*$")
    rows, cur = [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        m = header.match(line)
        if m:
            cur = dict(state=m.group(1), county=m.group(2), year=year,
                       total_r=to_number(m.group(4)),
                       total_e=to_number(m.group(5)),
                       same_r=0.0, same_e=0.0, for_r=0.0, for_e=0.0)
            continue
        m = nonmig.match(line)
        if m and cur is not None and m.group(2) == cur["county"]:
            cur["non_r"] = to_number(m.group(3))
            cur["non_e"] = to_number(m.group(4))
            rows.append(cur)
            cur = None
            continue
        if cur is not None:
            m = dest.match(line)
            if m:
                ds, r, e = m.group(1), to_number(m.group(5)) or 0, \
                           to_number(m.group(6)) or 0
                if ds == cur["state"]:
                    cur["same_r"] += r
                    cur["same_e"] += e
                elif ds in FOREIGN:
                    cur["for_r"] += r
                    cur["for_e"] += e
    return rows


def parse_xls(path, year):
    """1992-93 through 2003-04 per-state workbooks, all counties."""
    book = xlrd.open_workbook(path, on_demand=True)
    name = book.sheet_names()[0]
    for n in book.sheet_names():
        if "outflow" in n.lower():
            name = n
            break
    sheet = book.sheet_by_name(name)
    counties = {}
    for r in range(sheet.nrows):
        row = [str(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
        if len(row) < 8:
            continue
        st = clean_code(row[0])
        cty = clean_code(row[1]).zfill(3)
        if not st.isdigit() or not cty.isdigit() or cty == "000":
            continue
        ds = clean_code(row[2]).zfill(2)
        sub = clean_code(row[3]).zfill(3)
        r_, e_ = to_number(row[6]), to_number(row[7])
        key = (st, cty)
        if key not in counties:
            counties[key] = dict(state=st.zfill(2), county=cty, year=year,
                                 total_r=None, total_e=None, same_r=0.0,
                                 same_e=0.0, for_r=0.0, for_e=0.0,
                                 non_r=None, non_e=None)
        cur = counties[key]
        if ds == "00":
            cur["total_r"], cur["total_e"] = r_, e_
        elif ds == "63" and sub == "050":
            cur["non_r"], cur["non_e"] = r_, e_
        elif ds == "63":
            continue
        elif ds == st.zfill(2):
            cur["same_r"] += r_ or 0
            cur["same_e"] += e_ or 0
        elif ds in FOREIGN:
            cur["for_r"] += r_ or 0
            cur["for_e"] += e_ or 0
    return list(counties.values())



def parse_xls_summary(path, year):
    """1995-2003 workbooks: summary codes 96 / 97-0 / 97-1 / 97-3 / 98 and
    a labelled Non-Migrants row (ported from 01's clean_irs_xls)."""
    book = xlrd.open_workbook(path)
    name = book.sheet_names()[0]
    for n in book.sheet_names():
        if "outflow" in n.lower():
            name = n
            break
    sheet = book.sheet_by_name(name)
    counties = {}
    for r in range(sheet.nrows):
        row = [str(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
        if len(row) < 8:
            continue
        st = clean_code(row[0])
        cty = clean_code(row[1])
        if not st.isdigit() or cty in ("", "0") or not cty.isdigit():
            continue
        ds = clean_code(row[2])
        sub = clean_code(row[3])
        label = str(row[5]).lower()
        r_, e_ = to_number(row[6]), to_number(row[7])
        key = (st.zfill(2), cty.zfill(3))
        cur = counties.setdefault(key, dict(
            state=key[0], county=key[1], year=year, total_r=None,
            total_e=None, same_r=None, same_e=None, for_r=None, for_e=None,
            non_r=None, non_e=None, diff_r=None, diff_e=None))
        if "non-migrant" in label:
            cur["non_r"], cur["non_e"] = r_, e_
        elif ds == "96":
            cur["total_r"], cur["total_e"] = r_, e_
        elif ds == "97" and sub in ("0", "000"):
            pass
        elif ds == "97" and sub in ("1", "001"):
            cur["same_r"], cur["same_e"] = r_, e_
        elif ds == "97" and sub in ("3", "003"):
            cur["diff_r"], cur["diff_e"] = r_, e_
        elif ds == "98":
            cur["for_r"], cur["for_e"] = r_, e_
    return list(counties.values())

records = []
for year in range(1990, 2004):
    folder = RAW / f"{year}-{year + 1}" / \
        f"{year}to{year + 1}CountyMigrationOutflow"
    if not folder.exists():
        print(f"  !! {folder} missing — skipped")
        continue
    files = sorted(folder.iterdir())
    n0 = len(records)
    for f in files:
        if f.suffix.lower() == ".txt":
            records += parse_txt(f, year)
        elif f.suffix.lower() in (".xls", ".xlsx"):
            records += (parse_xls_summary(f, year) if year >= 1995
                        else parse_xls(f, year))
    print(f"{year}: {len(files)} files -> {len(records) - n0} county rows")

df = pd.DataFrame(records)
df["state"] = df["state"].str.zfill(2)
df["fips"] = df["state"] + df["county"]
# Kalawao into Maui, as everywhere else in the project.
df.loc[df.fips == "15005", "fips"] = "15009"
num = [c for c in ["total_r", "total_e", "same_r", "same_e", "for_r",
       "for_e", "non_r", "non_e", "diff_r", "diff_e"] if c in df.columns]
df = df.groupby(["fips", "year"], as_index=False)[num].sum(min_count=1)

resid = (df.total_r - df.same_r - df.for_r).clip(lower=0)
df["diff_r"] = df.get("diff_r", resid)
df["diff_r"] = df["diff_r"].fillna(resid)
df["households_at_risk"] = df.non_r + df.total_r
df["out_rate_all"] = df.total_r / df.households_at_risk
df["out_rate_diff_state"] = df.diff_r / df.households_at_risk
df["out_rate_same_state"] = df.same_r / df.households_at_risk

df = df.rename(columns={
    "total_r": "total_returns", "total_e": "total_exemptions",
    "same_r": "same_state_returns", "same_e": "same_state_exemptions",
    "for_r": "foreign_returns", "for_e": "foreign_exemptions",
    "non_r": "nonmigrant_returns", "non_e": "nonmigrant_exemptions",
    "diff_r": "diff_state_returns"})
df.to_csv(OUT, index=False)

print("\n#lemme seee it")
print(f"\nWritten to {OUT}")
print(f"Rows: {len(df):,}  counties: {df.fips.nunique():,}  "
      f"years: {df.year.min()}-{df.year.max()}")

# Cross-check against the Hawaii panel built by 01 from the same vintages.
hi_old = pd.read_csv("data/processed/irs_hawaii_migration_1990_2023.csv")
hi_old = hi_old[(hi_old.direction == "out") & (hi_old.year <= 2003)]
fmap = {"Hawaii County": "15001", "Honolulu County": "15003",
        "Kauai County": "15007", "Maui County": "15009"}
merged = hi_old.assign(fips=hi_old.county.map(fmap)).merge(
    df, on=["fips", "year"], suffixes=("_hi", "_nat"))
delta = (merged["diff_state_returns_hi"]
         - merged["diff_state_returns_nat"]).abs()
print(f"\nHawaii cross-check 1990-2003 (diff-state out returns): "
      f"max abs difference = {delta.max():.1f}, "
      f"NaN mismatches = {int((merged.diff_state_returns_hi.notna() != merged.diff_state_returns_nat.notna()).sum())} "
      f"across {len(merged)} county-years")
print("\nKauai out_rate_diff_state around Iniki (Sept 1992):")
print(df[df.fips == "15007"].set_index("year")
      .loc[1990:1996, "out_rate_diff_state"].round(4).to_string())
