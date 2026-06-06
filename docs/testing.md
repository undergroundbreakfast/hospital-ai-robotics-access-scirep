# Paper 1 Validation Test Suite

This repository includes a pytest suite for validating the released Paper 1
code package and committed result artifacts.

The suite has three layers:

1. **Release-package checks** verify required files are present, the SHA-256
   manifest matches the current files, and restricted raw data are not committed.
2. **Result-regression checks** validate that committed result artifacts contain
   the key values reported in the manuscript and Supplementary Information,
   including primary county estimates, hospital confirmatory endpoints,
   access-inequality summaries, pre-exposure checks, system-membership
   diagnostics, organizational-capacity diagnostics, overlap weighting, and
   exposure-misclassification sensitivity.
3. **Optional database-contract checks** validate that the local PostgreSQL
   database still exposes the source tables/views and columns used by the
   reproducibility scripts.

## Run Offline Artifact Tests

```sh
python -m pip install -r requirements-dev.txt
pytest
```

These tests do not require the licensed AHA source files or database access.
They validate the public result artifacts committed to this repository.

## Run Database Contract Tests

Database tests are skipped by default because they require a local PostgreSQL
database containing the licensed/non-public source data.

Set connection variables, then opt in with `RUN_DB_TESTS=1`:

```sh
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=Research_TEST
export PGUSER=postgres
# Set PGPASSWORD or POSTGRESQL_KEY in your shell or secret manager.

RUN_DB_TESTS=1 pytest -m db
```

The database tests are intentionally contract-level checks. They verify required
objects, columns, and lightweight source-row presence without running the full
analysis pipeline or scanning large views.

The database contract checks the focused AHA 2024 extract used by the primary
manuscript workflow and the broader `aha_survey_data` table that contains the
raw organizational-capacity fields used in Reviewer 3 diagnostics (`ceamt`,
`crnfte`, `ftern`, `gfeet`). These fields are not all present in every AHA
extract table, so tests validate them against `aha_survey_data`.

## What This Suite Does Not Prove

The default tests are artifact-regression checks. A passing run means the
committed CSVs, figures, scripts, SQL files, and manifest are internally
consistent with the curated key-result values in this release. It does not prove
that the licensed AHA raw-data pipeline was rerun end to end, and it does not
independently parse the manuscript PDF or LaTeX source. Manuscript-to-artifact
binding is maintained by curated key-result CSVs and regression tests, not by
automatic extraction from the submitted document.

## What A Failure Means

- A **result-regression failure** usually means a committed or regenerated CSV
  no longer matches the curated manuscript/SI value in this release. Either the
  manuscript needs updating, the curated key-result artifact needs updating, or
  the analysis output changed unexpectedly.
- A **database-contract failure** usually means a table/view/column was renamed,
  dropped, or not loaded into the local database.
- A **manifest failure** means files changed after `FILE_MANIFEST_SHA256.txt`
  was generated. Regenerate the manifest after intentional edits.
