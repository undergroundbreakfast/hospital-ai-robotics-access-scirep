# Reproducibility Scope

This package is a public release companion for the Scientific Reports revision. It is designed to support auditability without redistributing licensed AHA hospital-level survey records.

## Included

- Python scripts used for the outcome, access, moderation, and revision-diagnostic analyses.
- SQL view definitions that document the linked PostgreSQL analytic layer.
- Manuscript figures and summary tables.
- County-level aggregate derived data for the organizational-capacity sensitivity analysis.
- Public-data source notes and manifests.

## Excluded

- Raw or row-level AHA Annual Survey files.
- Hospital-level AHA-derived analytic datasets.
- Raw CMS archive files; these can be downloaded from official CMS URLs using the included script.
- Local database credentials, paths, caches, or virtual environments.

## Manuscript Mapping

- Table S3: `results/tables/revision_diagnostics/table_s3_pre_exposure_balance_results.csv`
- Table S27: `results/tables/revision_diagnostics/table_s27_system_membership_balance.csv` and `table_s27_system_membership_aipw_sensitivity.csv`
- Table S28: `results/tables/revision_diagnostics/table_s28_organizational_capacity_balance.csv` and `table_s28_organizational_capacity_aipw_sensitivity.csv`
- Main Figure 1: `results/figures/figure1_access_inequality_three_panel.png`
- Main Figure 2: `results/figures/figure2_dv21_forest_plot.png`
