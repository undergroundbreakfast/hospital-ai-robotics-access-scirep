# Suggested Run Order

1. Load licensed AHA files into PostgreSQL using the local schema expected by the SQL views.
2. Load public CMS, CDC WONDER, County Health Rankings, and HCRIS inputs into PostgreSQL.
3. Apply the SQL views in `sql/views/`.
4. Rebuild the main outcome/access outputs:

```sh
python code/replicate_scirep_outcomes.py
python code/geospatial_access_lorenz.py
python code/generate_moderation_plots.py
```

5. Rebuild reviewer-response diagnostics:

```sh
python code/support/build_cms_2022_quality_extract.py
psql "$DATABASE_URL" \
  -v cms_2022_csv="data/public/cms_2022_hospital_quality/filtered/cms_2022_hospital_quality_long.csv" \
  -f sql/load_cms_2022_hospital_quality.psql.sql
python code/support/build_pre_exposure_balance.py
python code/support/load_hcris_2023_financial.py
python code/support/build_system_membership_sensitivity.py
python code/support/build_organizational_capacity_sensitivity.py
```
