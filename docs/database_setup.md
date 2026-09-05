# Database setup

Use a dedicated PostgreSQL database. Do not apply setup scripts to an unrelated production database. The analysis runs use SELECT queries; the setup and CMS/HCRIS loaders intentionally create/update local database objects.

## Input contracts

`sql/source_schema.json` and `sql/source_tables.sql` describe the named, typed **prepared input tables** expected by the analysis. They contain no records, owner statements, grants, or credentials. The 2026-09-05 recovery adds the missing view dependencies and `safe_to_numeric` function. Original v1.0.4 view bodies are retained, with identifier quoting corrected for the mixed-case YPLL view.

The source tables include:

| Family | Required preparation |
|---|---|
| AHA annual files | Licensed 2022/2023/2024 extracts, the study-year `aha_survey_data`, ownership/division code tables, and AHA/CMS crosswalk; retain source IDs as text. |
| CMS hospital outcomes | Study-period hospital outcome extracts, hospital general information, and corresponding 2019 baseline files. The `nber_*` and `f_2019_nber_*` names are local staging names, not CMS download filenames. |
| CHR and county context | CHR analytic chunks, 2019 YPLL, HRSA county context, Medicaid expansion, and county identifiers. |
| CDC and population | 2023 age-banded deaths, hospital-place-of-death counts, COVID deaths, and age-specific Census population denominators. Preserve suppression strings for the supplied transformations. |
| Geography/crosswalks | Hospital coordinates and county crosswalks used by the study. Some geocodes are part of licensed inputs. |

This package supplies the downstream transformation layer. It does **not** supply a universal importer for every vendor file or a byte-for-byte archived snapshot of all raw sources. The initial source-to-staging extraction/normalization must follow the column contracts and study measurement windows. Do not fill absent inputs with invented observations. This remaining source-ingestion boundary is explicit so that the presence of runnable SQL is not mistaken for a complete raw-download pipeline.

## Build sequence

With PGHOST/PGPORT/PGUSER/PGDATABASE configured (psql uses PGPASSWORD):

```sh
psql -v ON_ERROR_STOP=1 -f sql/source_tables.sql
# Load the prepared source tables using their matching columns and study-year data.
psql -v ON_ERROR_STOP=1 -f sql/build_views.psql.sql
```

The first command creates empty tables if absent; it does not load data or replace existing table definitions. The second builds all 31 views in dependency order inside one transaction. A failure rolls back the view build. The same commands are exercised against an empty PostgreSQL 16 database in CI, verifying dependencies and types rather than analytical results.

For CMS 2022 snapshots and HCRIS, use the included download/processing commands in `run_order.md`. The HCRIS loader resolves an official current endpoint; its generated source URL and download date must be retained with any rerun. Different releases of source data may change results.

## Optional contract check

```sh
RUN_DB_TESTS=1 python -m pytest -m db
```

These checks verify schema and selected source presence. They are not full statistical reproduction. Run them only against the prepared study database; an empty CI schema is expected to have zero source rows.
