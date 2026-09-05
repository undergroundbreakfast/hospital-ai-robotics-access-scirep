# CMS 2022 Hospital Quality Extracts

This folder stages official CMS Provider Data Catalog hospital archive snapshots
for 2022 and filters them to the hospital-level measures needed for revision
pretrend analyses:

- `SEP_1`: Appropriate care for severe sepsis and septic shock
- `MORT_30_PN`: Death rate for pneumonia patients

## Official Source Archives

- January 2022: https://data.cms.gov/provider-data/sites/default/files/archive/Hospitals/2022/hospitals_01_2022.zip
- April 2022: https://data.cms.gov/provider-data/sites/default/files/archive/Hospitals/2022/hospitals_04_2022.zip
- July 2022: https://data.cms.gov/provider-data/sites/default/files/archive/Hospitals/2022/hospitals_07_2022.zip
- October 2022: https://data.cms.gov/provider-data/sites/default/files/archive/Hospitals/2022/hospitals_10_2022.zip

## Outputs

- `archives/`: downloaded official CMS zip archives; not committed here.
- `raw/`: raw CSV copies extracted from each archive; not committed here:
  - `Timely_and_Effective_Care-Hospital.csv`
  - `Complications_and_Deaths-Hospital.csv`
  - `Hospital_General_Information.csv`
  - `Measure_Dates.csv`
- `filtered/cms_2022_hospital_quality_long.csv`: combined Postgres-ready long file; rebuild locally with the script below.
- `filtered/cms_2022_sep1_hospital_long.csv`: SEP-1 rows only; rebuild locally.
- `filtered/cms_2022_mort30pn_hospital_long.csv`: pneumonia mortality rows only; rebuild locally.
- `cms_2022_extract_manifest.csv`: row counts and nonmissing score counts by snapshot and measure; committed here.
- `sql/load_cms_2022_hospital_quality.psql.sql`: load script for Postgres via `psql`; see repository root.

## Extract Manifest

| Snapshot | Measure | Rows | Nonmissing scores |
|---|---:|---:|---:|
| 2022_01 | SEP_1 | 4,711 | 2,949 |
| 2022_01 | MORT_30_PN | 4,848 | 3,896 |
| 2022_04 | SEP_1 | 4,711 | 2,949 |
| 2022_04 | MORT_30_PN | 4,848 | 3,896 |
| 2022_07 | SEP_1 | 4,706 | 3,059 |
| 2022_07 | MORT_30_PN | 4,843 | 0 |
| 2022_10 | SEP_1 | 4,708 | 3,069 |
| 2022_10 | MORT_30_PN | 4,845 | 0 |

The July and October 2022 `MORT_30_PN` hospital rows are present in the CMS
files, but their scores are marked not available. The January and April 2022
snapshots contain populated pneumonia mortality scores.

## Loading Into Postgres

After rebuilding the filtered long CSV, load it from the repository root:

```sh
psql "YOUR_CONNECTION_STRING" \
  -v cms_2022_csv="data/public/cms_2022_hospital_quality/filtered/cms_2022_hospital_quality_long.csv" \
  -f "sql/load_cms_2022_hospital_quality.psql.sql"
```

The script creates or updates:

- `public.cms_2022_hospital_quality_long`
- `public.vw_cms_2022_hospital_quality_scored`

The load is idempotent on `(snapshot, measure_id, facility_id)` and uses
`ON CONFLICT DO UPDATE`, so rerunning the script refreshes rows rather than
duplicating them.

## Rebuilding

After replacing or refreshing the archives, rebuild the filtered CSVs with:

```sh
python code/support/build_cms_2022_quality_extract.py --download
```

The script resolves all paths from the repository root. Omit `--download` when the archives are already present.
