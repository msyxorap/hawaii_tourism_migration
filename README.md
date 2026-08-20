# hawaii_tourism_migration

Data and code for Hawaii Migration Dissertation — a county-year panel analysis of
tourism exposure and out-migration, combining a Hawaii panel (1990–2022)
with a national county panel (2011–2022, extended to 1990 for the
natural-experiment analysis).

## Data sources (data/raw/)

| Folder | File(s) | Source | Downloaded |
|---|---|---|---|
| irs.zip | Hawaii state extracts | IRS SOI Migration Data, Hawaii county files, 1990–91 through 2022–23. irs.gov/statistics/soi-tax-stats-migration-data | Aug 14–15, 2026 |
| irs_national/ | per-year folders, 1990-1991 … 2022-2023 | IRS SOI Migration Data, national county-to-county inflow/outflow, all formats (txt 1990–92, per-state xls 1992–2011, national csv 2011–23). Same page as above | Aug 18, 2026 |
| dbedt.zip | Annual visitor report tables | DBEDT Annual Visitor Research Reports, supporting tables, 1999–2024. dbedt.hawaii.gov/economic/tourism/annual-reports | Aug 14, 2026 |
| laus.zip | laucnty90–laucnty25 | BLS Local Area Unemployment Statistics, annual county files. bls.gov/lau/tables.htm | Aug 15, 2026 |
| fhfa.zip | hpi_at_county.xlsx | FHFA Annual House Price Index, county (developmental), all-transactions. fhfa.gov/data/hpi/datasets | Aug 15, 2026 |
| qcew/ | 1975.zip … 2025.zip | BLS Quarterly Census of Employment and Wages, annual by-industry files. bls.gov/cew/downloadable-data-files.htm | Aug 17–18, 2026 |

Raw files are committed exactly as downloaded and are never edited, with
one documented exception: the QCEW year-zips contain the two industry CSVs
("Total, all industries" and "72 … Accommodation and food services")
extracted unmodified from BLS's annual by-industry archives and re-zipped
per year, because the full archives exceed repository size limits. Files
in data/processed/ are generated only by scripts in Code/.

## Pipeline (Code/)

01–04 parse the Hawaii sources (IRS, DBEDT, LAUS, FHFA); 05 builds the
Hawaii county-year panel; 06–07 estimate the Hawaii regressions and the
small-cluster robustness program; 08 parses QCEW into the national
tourism-employment share; 09 builds the national panel; 10 estimates the
national regressions; 11 runs the exit-geography contrast; 12 the COVID
synthetic control (Maui); 13 figures and descriptives; 14 parses the
early-format national IRS files (1990–2003); 15 the Hurricane Iniki
natural experiment; 16 the 2016 measurement-artifact diagnostics. Each
script prints a verification report; all were confirmed to produce
identical output on two independent machines.

## Known data notes

- IRS series breaks: 2011–12 (new methodology) and 2022–23 (updated
  matching, ~5% more returns; revised suppression rules). See
  docs/2223inpublicmigdoc.pdf.
- IRS mid-2010s artifact: gross migrant counts are inflated in the
  2015–16 and 2016–17 vintages and fall sharply from 2018–19 under
  updated processing; diagnostics and regime-split robustness in
  Code/16_spike_2016.py and output/note_2016_spike.txt.
- IRS 1990–91 and 1991–92 text files omit smaller counties entirely
  (source limitation, not parsing loss).
- IRS negative entries denote disclosure suppression and are treated as
  missing, never as zeros. Kalawao County (15005) is folded into Maui
  (15009) throughout.
- DBEDT 1999 and 2003 values sourced from prior-year columns of the 2000
  and 2004 report tables (those editions were PDF-only).
- QCEW NAICS-based files for 1990–2000 are BLS reconstructions from the
  SIC classification. County-industry cells failing BLS disclosure rules
  are suppressed (absent in older vintages, zero-with-flag in newer) and
  treated as missing; Kauai's accommodation sector is suppressed from the
  mid-2000s onward.
