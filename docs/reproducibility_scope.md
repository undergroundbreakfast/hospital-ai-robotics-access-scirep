# Reproducibility scope

The package provides analysis code, SQL transformations, archived aggregate results, publication figures, and one county-level aggregate dataset. Licensed AHA records and generated hospital-level analytic data are excluded.

See `database_setup.md` for source staging and recovered SQL, `geographic_inputs.md` for geographic prerequisites, `run_order.md` for generated-to-archived output mapping, and `result_provenance.md` for numerical verification limits.

The CMS 2022 extractor downloads the four fixed official archives only with `--download`; otherwise it uses local archives. Other raw source ingestion is not automated universally. Offline tests are not end-to-end replication. An empty-schema CI build validates SQL dependencies, not the scientific estimates.
