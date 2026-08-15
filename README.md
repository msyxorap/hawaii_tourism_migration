# hawaii_tourism_migration

Data and code for [dissertation title] — a county-year panel analysis of
tourism exposure and out-migration in Hawaii, 1990–2024.

## Data sources (data/raw/)

| Folder | File | Source | Downloaded |
|--------|------|--------|------------|
| irs/ | County-to-county_data.zip | IRS SOI Migration Data, county-to-county inflow/outflow, 1990–91 through 2022–23. irs.gov/statistics/soi-tax-stats-migration-data | Aug 14–15, 2026 |
| dbedt/ | Annual_Visitor_data.zip | DBEDT Annual Visitor Research Reports, supporting tables, 1999–2024. dbedt.hawaii.gov/economic/tourism/annual-reports | Aug 14, 2026 |
| laus/ | County_Unemployment_Data.zip | BLS Local Area Unemployment Statistics, annual county files (laucnty90–laucnty25). bls.gov/lau/tables.htm | Aug 15, 2026 |
| fhfa/ | hpi_at_county.xlsx | FHFA Annual House Price Index, county (developmental), all-transactions. fhfa.gov/data/hpi/datasets | Aug 15, 2026 |

Raw files are committed exactly as downloaded and are never edited.
Files in data/processed/ are generated only by scripts in code/.

## Known data notes
- IRS series breaks: 2011–12 (new methodology) and 2022–23 (updated
  matching, ~5% more returns; revised suppression rules). See
  docs/2223inpublicmigdoc.pdf.
- DBEDT 1999 and 2003 values sourced from prior-year columns of the
  2000 and 2004 report tables (those editions were PDF-only).
