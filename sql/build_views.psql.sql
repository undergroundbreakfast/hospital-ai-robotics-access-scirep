\set ON_ERROR_STOP on
BEGIN;
\ir functions/public.safe_to_numeric.sql
\ir views/public.vw_medicaid_expansion.sql
\ir views/public.vw_2023_hospital_deaths.sql
\ir views/public.vw_tech_enabled_beds_by_county_adjpd.sql
\ir views/public.vw_hospital_ai_score_adjpd.sql
\ir views/public.vw_robotics_score_by_hospital_adjpd.sql
\ir views/public.vw_conceptual_model_adjpd.sql
\ir views/public.vw_hospital_performance_metrics.sql
\ir views/public.vw_healthcare_associated_infections.sql
\ir views/public.vw_hospital_patient_experience.sql
\ir views/public.vw_unplanned_hospital_visits.sql
\ir views/public.vw_timely_effective_care.sql
\ir views/public.vw_hospital_level_aipw.sql
\ir views/public.vw_2023_age_adj_YPLL.sql
\ir views/public.vw_hospital_county_bridge.sql
\ir views/public.vw_irr_cdc_wonder_mortality_2023.sql
\ir views/public.vw_geocoded_hospitals.sql
\ir views/public.combined_chr_analytic.sql
\ir views/public.vw_conceptual_model_variables_adjpd.sql
\ir views/public.vw_adjpd_weighted_tech_summary.sql
\ir views/public.vw_county_tech_summary_adjpd.sql
\ir views/public.vw_irr_total_margin.sql
\ir views/public.hospital_level_placebo_2019.sql
\ir views/public.vw_hospital_aipw_with_placebo.sql
\ir views/public.vw_capex_per_sqfoot.sql
\ir views/public.vw_tech_enabled_beds_by_county.sql
\ir views/public.vw_hospital_ai_score.sql
\ir views/public.vw_robotics_score_by_hospital.sql
\ir views/public.vw_conceptual_model_variables.sql
\ir views/public.vw_county_covid_deaths.sql
\ir views/public.vw_county_file_export_wide.sql
\ir views/public.vw_hospital_ownership_percentages.sql
COMMIT;
