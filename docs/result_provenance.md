# Publication result provenance and proof reconciliation

The v1.0.4 archive was compared with the local revision-support output saved on 2026-05-25 and the accepted manuscript's Table 1 on 2026-09-05. No numerical estimates were changed during package repair.

## CT6

For MO14 → CT6, both the saved revision-support file and the archived `primary_county_crossfit_summary.csv` contain:

- N = 2,896; treated = 414; controls = 2,482.
- Cross-fitted adjusted difference = −25.492858508367927.
- 95% CI = [−37.79750234540421, −13.188214671331641].
- ESS = 413.00842277059695; p = 0.000048916686447908475.

The publisher proof instead prints [−39.7, −11.3] and ESS 409.1 alongside the matching −25.5 estimate. Those values should be reconciled in the proof; the archived outputs support rounded CI **[−37.8, −13.2]** and **ESS 413.0**. ESS 409.1 belongs to MO14 → DV21 in the same output file. The repo now tests CT6's CI and ESS explicitly. This corroboration is from saved output, not an independent fresh fit of the licensed-data model.

## Hospital denominators

`main_hospital_confirmatory_results.csv` is a curated registry of the manuscript's confirmatory results. `source_frame_n=6166` denotes the linked AHA hospital frame, while `model_n=2272` for SEP-1 and `model_n=2795` for pneumonia denote the outcome-and-2019-baseline samples in accepted Table 1. A read-only aggregate check of those pairs in the local hospital baseline view on 2026-09-05 returned the same endpoint counts. It did not refit either model or prove that all downstream exclusions were identical.

The prior generic `n=6166` field was replaced by these explicit fields; tests no longer label the source frame as a model denominator. Estimates, intervals, and relative percentages are unchanged.

## Scope of verification

The offline tests assert selected archived numerical values, cross-table consistency, manifest integrity, connection configuration, input paths, and SQL setup. They do not derive the paper's estimates from raw data. The original computational environment and all intermediate source-ingestion scripts were not frozen in v1.0.4. The newly pinned Python 3.11 environment and recovered SQL improve executability, but do not establish an end-to-end numerical rerun. Retain that distinction in availability statements and release notes.
