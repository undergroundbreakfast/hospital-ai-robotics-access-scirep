# Analysis run order and output map

First install the Python 3.11 analysis requirements and follow `database_setup.md` and `geographic_inputs.md`. All commands run from the repository root. Keep credentials outside Git.

```sh
python code/support/build_cms_2022_quality_extract.py --download
psql -v ON_ERROR_STOP=1 -v cms_2022_csv="data/public/cms_2022_hospital_quality/filtered/cms_2022_hospital_quality_long.csv" -f sql/load_cms_2022_hospital_quality.psql.sql
python code/support/load_hcris_2023_financial.py
python code/replicate_scirep_outcomes.py
python code/geospatial_access_lorenz.py
python code/generate_moderation_plots.py
python code/support/build_pre_exposure_balance.py
python code/support/build_system_membership_sensitivity.py
python code/support/build_organizational_capacity_sensitivity.py
```

Omit `--download` to reuse the fixed CMS archives already in the documented archive directory. The extraction writes into that same public-data tree, not beside the Python script. Raw/filtered snapshots are ignored by Git.

| Generated output | Archived publication counterpart |
|---|---|
| `outputs/pre_exposure/pre_exposure_balance_results.csv` | `results/tables/revision_diagnostics/table_s3_pre_exposure_balance_results.csv` |
| `outputs/pre_exposure/aha_2022_to_2023_transition_counts.csv` | `results/tables/revision_diagnostics/pre_exposure_transition_counts.csv` |
| `outputs/system_membership/system_membership_balance.csv` | `results/tables/revision_diagnostics/table_s27_system_membership_balance.csv` |
| `outputs/system_membership/system_membership_aipw_sensitivity.csv` | `results/tables/revision_diagnostics/table_s27_system_membership_aipw_sensitivity.csv` |
| `outputs/organizational_capacity/organizational_capacity_balance.csv` | `results/tables/revision_diagnostics/table_s28_organizational_capacity_balance.csv` |
| `outputs/organizational_capacity/organizational_capacity_aipw_sensitivity.csv` | `results/tables/revision_diagnostics/table_s28_organizational_capacity_aipw_sensitivity.csv` |
| `outputs/organizational_capacity/organizational_capacity_county_dataset.csv` | `data/derived/county/organizational_capacity_county_dataset.csv` |

Support Markdown/LaTeX summaries stay with generated output. Hospital-level analytic exports go to `data/restricted/` and must not be promoted into the public archive. Main outcome and moderation scripts create timestamped run folders; mapping files use the ignored `code/figures/` and `code/publication_outputs/` directories. Compare generated aggregate outputs before explicitly copying any to `results/`; the script does not silently overwrite the cited result archive.

Archived `main_hospital_confirmatory_results.csv` and `access_inequality_key_results.csv` are curated publication registries. They are checked against documented values rather than emitted directly by those workflows. See `result_provenance.md`.
