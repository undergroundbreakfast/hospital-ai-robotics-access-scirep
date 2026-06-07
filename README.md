# Hospital AI/Robotics Access and Outcomes - Scientific Reports Reproducibility Package

This repository contains reproducibility code and redistributable artifacts for the revised Scientific Reports manuscript:

**Hospital AI and Robotics Adoption, Access Inequality, and County Mortality: A National Study Across 3,143 U.S. Counties**

The package is intentionally scoped for public release. It includes analysis scripts, SQL view definitions, derived aggregate county-level artifacts, manuscript figures, and summary tables. It does **not** include licensed American Hospital Association (AHA) Annual Survey hospital-level source files or hospital-level AHA-derived analytic datasets.

## Repository Contents

- `code/`: Python workflows for the main county/hospital analyses, geospatial access and Lorenz/Gini calculations, moderation plots, and reviewer-response diagnostics.
- `sql/views/`: SQL view definitions used to build the linked analytic tables in PostgreSQL.
- `results/tables/`: generated summary tables and sensitivity-analysis outputs used in the manuscript and Supplementary Information.
- `results/tables/revision_diagnostics/`: reviewer-response outputs for the pre-exposure CMS check, system-membership diagnostic, and organizational-capacity diagnostic.
- `results/figures/`: manuscript and supplementary figure artifacts.
- `data/derived/county/`: redistributable county-level aggregate analytic file for the organizational-capacity sensitivity analysis.
- `data/public/`: source notes and manifests for public CMS/HCRIS inputs.
- `docs/`: short notes mapping reviewer-response diagnostics to manuscript tables.

## Data Availability Boundaries

The AHA Annual Survey data were used under a third-party data-use license and cannot be redistributed here. The code assumes that licensed AHA files have been loaded into a local PostgreSQL database under the table/view names referenced in `sql/views/` and the scripts.

Public data sources used by the workflows include:

- CMS Provider Data hospital outcomes: <https://data.cms.gov/provider-data/topics/hospitals>
- CDC WONDER mortality data: <https://wonder.cdc.gov/>
- County Health Rankings & Roadmaps: <https://www.countyhealthrankings.org/health-data>
- CMS HCRIS cost report public-use files: <https://www.cms.gov/data-research/statistics-trends-and-reports/cost-reports>

## Environment

Python 3.11 was used for the revision analyses. Install the core dependencies with:

```sh
python -m pip install -r requirements.txt
```

Database connection settings are read from environment variables:

```sh
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=Research_TEST
export PGUSER=postgres
# Set PGPASSWORD or POSTGRESQL_KEY in your shell or secret manager.
```

The scripts also accept `POSTGRESQL_KEY` as a password environment variable for compatibility with the author's local workflow.

## Main Workflows

Run from the repository root after the PostgreSQL database has been prepared:

```sh
python code/replicate_scirep_outcomes.py
python code/geospatial_access_lorenz.py
python code/generate_moderation_plots.py
python code/support/build_pre_exposure_balance.py
python code/support/build_system_membership_sensitivity.py
python code/support/build_organizational_capacity_sensitivity.py
```

The CMS 2022 public-reporting snapshots used for the pre-exposure balance check can be rebuilt with:

```sh
python code/support/build_cms_2022_quality_extract.py
psql "$DATABASE_URL" \
  -v cms_2022_csv="data/public/cms_2022_hospital_quality/filtered/cms_2022_hospital_quality_long.csv" \
  -f sql/load_cms_2022_hospital_quality.psql.sql
```

## Validation Tests

The repository includes a pytest suite that validates the release package and
checks that committed result artifacts contain the key values reported in the
manuscript and Supplementary Information:

```sh
python -m pip install -r requirements-dev.txt
pytest
```

Optional database contract checks verify that the local PostgreSQL database has
the expected source objects and columns:

```sh
RUN_DB_TESTS=1 pytest -m db
```

See `docs/testing.md` for scope and interpretation. These tests are artifact
regression and database-contract checks. They do not rerun the licensed AHA
pipeline end to end, and they do not independently parse the manuscript PDF or
LaTeX source.

## Versioning

The intended journal-submission release tag is:

```sh
v1.0.3
```

Use the release tag that corresponds exactly to the submitted manuscript in the
Data Availability and Code Availability statements. If later test-suite,
artifact, or documentation corrections are included after the cited tag, create
and cite a new release tag rather than assuming an older tag contains those
changes.

## License

Code is released under the MIT License. Redistributable derived outputs are provided for scholarly reproducibility. Third-party source datasets remain subject to their original licensing terms.
