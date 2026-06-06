# Paper 1 Validation Test Suite

This repository includes a pytest suite for validating the released Paper 1
code and result artifacts.

The suite has three layers:

1. **Release-package checks** verify required files are present, the SHA-256
   manifest matches the current files, and restricted raw data are not committed.
2. **Result-regression checks** validate the numeric results cited in the
   manuscript and Supplementary Information, including primary county estimates,
   pre-exposure checks, system-membership diagnostics, organizational-capacity
   diagnostics, overlap weighting, and exposure-misclassification sensitivity.
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

The current local database contract includes one expected limitation: some raw
AHA fields used for full organizational-capacity reruns (`ceamt`, `crnfte`,
`ftern`, `gfeet`) may be absent from a slim AHA extract even though the released
analysis views and result artifacts are present. The corresponding test is
reported as `xfail` so this limitation remains visible without blocking artifact
validation.

## What A Failure Means

- A **result-regression failure** usually means a regenerated CSV no longer
  matches the manuscript/SI value. Either the manuscript needs updating or the
  analysis output changed unexpectedly.
- A **database-contract failure** usually means a table/view/column was renamed,
  dropped, or not loaded into the local database.
- A **manifest failure** means files changed after `FILE_MANIFEST_SHA256.txt`
  was generated. Regenerate the manifest after intentional edits.
