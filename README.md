# Hospital AI and robotics adoption and access inequality in the United States

Research code and aggregate artifacts accompanying the article accepted by **Scientific Reports** on 31 August 2026 (Johnson, Gefen, and Harrison).

**Publication companion: [v1.0.5](https://github.com/undergroundbreakfast/hospital-ai-robotics-access-scirep/releases/tag/v1.0.5).** The previously submitted version is [v1.0.4](https://github.com/undergroundbreakfast/hospital-ai-robotics-access-scirep/tree/v1.0.4). See [CHANGELOG.md](CHANGELOG.md) for package corrections and [result provenance](docs/result_provenance.md) for the CT6 proof discrepancy. Statistical result estimates have not been changed to match proof text.

Article DOI supplied in the publisher proof: [10.1038/s41598-026-70027-1](https://doi.org/10.1038/s41598-026-70027-1). The DOI may not resolve until publication. Citation metadata is in [CITATION.cff](CITATION.cff).

## What can be reproduced

The public package contains analysis code, SQL transformations, aggregate county data, and archived figures/results. It excludes licensed AHA hospital records and generated hospital-level analytic datasets. **Offline checks validate the archived artifacts and package; they do not rerun the study.** Full analysis requires licensed AHA data and the prepared public inputs described in [database setup](docs/database_setup.md) and [geographic inputs](docs/geographic_inputs.md).

The Python 3.11 dependency snapshot is a newly tested compatibility environment, not a recovered lockfile from the original analysis. The recovered SQL definitions are identified in [schema provenance](docs/recovered_schema.json). Building an empty schema is not evidence that a fresh source-data load reproduces all published estimates.

## Quick offline checks

From a clone of the repository, with Python 3.11:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest -q
```

These checks need no AHA data or database credentials. CI runs the artifact checks and builds the SQL schema in an empty PostgreSQL 16 service.

## Analysis environment and database

```sh
python -m pip install -r requirements.txt
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=Research_TEST
export PGUSER=postgres
# Set PGPASSWORD securely in the shell; do not commit it.
```

All workflows use these PG settings. The legacy `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRESQL_KEY` variables remain supported when the corresponding PG setting is absent. Standard PG settings take precedence. Passwords containing URL punctuation are supported.

Follow [database_setup.md](docs/database_setup.md) before running analyses; the scripts expect prepared input tables, not arbitrary raw files.

## Workflows and generated outputs

The ordered commands are in [docs/run_order.md](docs/run_order.md). Main workflows:

```sh
python code/replicate_scirep_outcomes.py
python code/geospatial_access_lorenz.py
python code/generate_moderation_plots.py
```

Support analyses write under ignored `outputs/`. Hospital-level analytic exports go under ignored `data/restricted/`. Mapping caches and source geography stay under ignored `code/shapefiles/` and related paths. Archived publication outputs in `results/` are updated only after comparison and review; [the output map](docs/run_order.md) identifies which generated files correspond to them.

## Repository contents

- `code/`: outcome, mapping, moderation, and support workflows.
- `sql/`: source-column contracts, required functions, dependency-ordered view setup, and the CMS loader.
- `results/`: archived aggregate results and manuscript figures.
- `data/derived/county/`: an aggregate county dataset for the capacity sensitivity analysis.
- `data/public/`: public-data source notes and extract manifests.
- `docs/`: setup, interpretation, and provenance notes.
- `tests/`: offline and optional database-contract checks.

## Data sources and boundaries

AHA Annual Survey files must be obtained under the relevant data-use agreement from [AHA Data](https://www.ahadata.com/aha-annual-survey-database). They are not redistributed here. Public inputs include [CMS hospital data](https://data.cms.gov/provider-data/topics/hospitals), [CDC WONDER](https://wonder.cdc.gov/), [County Health Rankings](https://www.countyhealthrankings.org/health-data), [CMS cost reports](https://www.cms.gov/data-research/statistics-trends-and-reports/cost-reports), and Census geography/population data.

## License and releases

Code is MIT licensed. Third-party data retain their source terms. The code license is separate from the publisher's article license. Cite the immutable release used for an analysis. Never move an existing cited tag to new content. Before tagging a new version, finalize citation metadata, regenerate `FILE_MANIFEST_SHA256.txt` with `python tools/update_manifest.py`, and run the tests.
