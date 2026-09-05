CREATE OR REPLACE VIEW public."vw_county_file_export_wide" AS
 SELECT vcm.county_fips,
    vcm.census_division,
    vcm.population,
    vcm.health_behaviors_score AS iv3_health_behaviors_score,
    vcm.social_economic_factors_score AS iv4_social_economic_factors_score,
    vcm.physical_environment_score AS iv2_physical_environment_score,
    vcm.medicaid_expansion_active AS iv1_medicaid_expansion_active,
    vcv.premature_death_raw_value AS dv21_premature_death_ypll_rate,
    vcm.health_outcomes_score AS dv2_health_outcomes_score,
    vcm.clinical_care_score AS dv1_clinical_care_score,
    vcm.avg_patient_services_margin AS dv3_avg_patient_services_margin,
    vcv.ratio_of_population_to_primary_care_physicians AS dv12_physicians_ratio,
    vcv.preventable_hospital_stays_raw_value AS dv15_preventable_stays_rate,
    vcm.weighted_ai_adoption_score AS mo1_genai_composite_score,
    vcm.weighted_robotics_adoption_score AS mo2_robotics_composite_score,
    vcts.pct_wfaiss_enabled_adjpd AS mo11_ai_staff_scheduling_pct,
    vcts.pct_wfaipsn_enabled_adjpd AS mo12_ai_predict_staff_needs_pct,
    vcts.pct_wfaippd_enabled_adjpd AS mo13_ai_predict_patient_demand_pct,
    vcts.pct_wfaiart_enabled_adjpd AS mo14_ai_automate_routine_tasks_pct,
    vcts.pct_wfaioacw_enabled_adjpd AS mo15_ai_optimize_workflows_pct,
    vcts.pct_robohos_enabled_adjpd AS mo21_robotics_in_hospital_pct,
    hrsa.irr_county_value AS sp5_irr_county_value,
    aha_agg.capex_sum AS fi1_capex_sum,
    aha_agg.adjpd_sum AS fi2_adjpd_sum,
        CASE
            WHEN aha_agg.adjpd_sum IS NOT NULL AND aha_agg.adjpd_sum <> 0::numeric THEN aha_agg.capex_sum / aha_agg.adjpd_sum
            ELSE NULL::numeric
        END AS fi_capex_intensity_ratio,
    aha_agg.n_federal_govt AS own_n_federal_govt,
    aha_agg.n_nonfederal_govt AS own_n_nonfederal_govt,
    aha_agg.n_not_for_profit AS own_n_not_for_profit,
    aha_agg.n_for_profit AS own_n_for_profit,
    aha_agg.n_hospitals_total AS own_n_hospitals_total,
    placebo.ypll_rate AS pl1_ypll_rate_2019,
    covid.covid_deaths_total AS cv1_covid_deaths_total,
    cdc.deaths_2023 AS ct4_cdc_deaths_2023,
    ypll2023.ct5_ypll_per_100k_low AS ct5_ypll_per_100k_low_2023,
    ypll2023.ct5_ypll_per_100k_mid AS ct5_ypll_per_100k_mid_2023,
    ypll2023.ct5_ypll_per_100k_high AS ct5_ypll_per_100k_high_2023,
    hospdeaths2023.total_age_adjusted_deaths AS ct6_hospital_deaths_age_adj_2023
   FROM vw_conceptual_model_adjpd vcm
     LEFT JOIN vw_conceptual_model_variables vcv ON vcm.county_fips = vcv.county_fips::text
     LEFT JOIN vw_adjpd_weighted_tech_summary vcts ON vcm.county_fips = vcts.county_fips
     LEFT JOIN ( SELECT lpad(TRIM(BOTH FROM hrsa_health_equity_data.county_fips_code::text), 5, '0'::text) AS county_fips,
            avg(
                CASE
                    WHEN NULLIF(TRIM(BOTH FROM hrsa_health_equity_data.irr_county_value::text), ''::text) ~ '^[0-9]+(\\.[0-9]+)?$'::text THEN hrsa_health_equity_data.irr_county_value::numeric
                    ELSE NULL::numeric
                END) AS irr_county_value
           FROM hrsa_health_equity_data
          GROUP BY (lpad(TRIM(BOTH FROM hrsa_health_equity_data.county_fips_code::text), 5, '0'::text))) hrsa ON vcm.county_fips = hrsa.county_fips
     LEFT JOIN ( SELECT lpad(TRIM(BOTH FROM aha_survey_data.fcounty::text), 5, '0'::text) AS county_fips,
            sum(
                CASE
                    WHEN NULLIF(TRIM(BOTH FROM aha_survey_data.ceamt::text), ''::text) ~ '^-?[0-9]+(\\.[0-9]+)?$'::text THEN aha_survey_data.ceamt::numeric
                    ELSE NULL::numeric
                END) AS capex_sum,
            sum(
                CASE
                    WHEN NULLIF(TRIM(BOTH FROM aha_survey_data.adjpd::text), ''::text) ~ '^-?[0-9]+(\\.[0-9]+)?$'::text THEN aha_survey_data.adjpd::numeric
                    ELSE NULL::numeric
                END) AS adjpd_sum,
            count(
                CASE
                    WHEN TRIM(BOTH FROM aha_survey_data.cntrl) = ANY (ARRAY['45'::text, '47'::text, '44'::text, '48'::text, '46'::text, '40'::text]) THEN 1
                    ELSE NULL::integer
                END) AS n_federal_govt,
            count(
                CASE
                    WHEN TRIM(BOTH FROM aha_survey_data.cntrl) = ANY (ARRAY['12'::text, '16'::text, '14'::text, '13'::text, '15'::text]) THEN 1
                    ELSE NULL::integer
                END) AS n_nonfederal_govt,
            count(
                CASE
                    WHEN TRIM(BOTH FROM aha_survey_data.cntrl) = ANY (ARRAY['23'::text, '21'::text]) THEN 1
                    ELSE NULL::integer
                END) AS n_not_for_profit,
            count(
                CASE
                    WHEN TRIM(BOTH FROM aha_survey_data.cntrl) = ANY (ARRAY['32'::text, '33'::text, '31'::text]) THEN 1
                    ELSE NULL::integer
                END) AS n_for_profit,
            count(*) AS n_hospitals_total
           FROM aha_survey_data
          WHERE aha_survey_data.fcounty IS NOT NULL
          GROUP BY (lpad(TRIM(BOTH FROM aha_survey_data.fcounty::text), 5, '0'::text))) aha_agg ON vcm.county_fips = aha_agg.county_fips
     LEFT JOIN chr_2019_ypll_placebo placebo ON vcm.county_fips = placebo.county_fips::text
     LEFT JOIN ( SELECT lpad(TRIM(BOTH FROM vw_county_covid_deaths.county_fips), 5, '0'::text) AS county_fips,
                CASE
                    WHEN NULLIF(TRIM(BOTH FROM vw_county_covid_deaths.deaths_involving_covid_19::text), ''::text) ~ '^[0-9]+(\\.[0-9]+)?$'::text THEN vw_county_covid_deaths.deaths_involving_covid_19::numeric
                    ELSE NULL::numeric
                END AS covid_deaths_total
           FROM vw_county_covid_deaths) covid ON vcm.county_fips = covid.county_fips
     LEFT JOIN ( SELECT lpad(TRIM(BOTH FROM cdc_2023_county_deaths_under_75yrs.county_fips::text), 5, '0'::text) AS county_fips,
            sum(
                CASE
                    WHEN NULLIF(TRIM(BOTH FROM cdc_2023_county_deaths_under_75yrs.deaths::text), ''::text) ~ '^[0-9]+(\\.[0-9]+)?$'::text THEN cdc_2023_county_deaths_under_75yrs.deaths::numeric
                    ELSE NULL::numeric
                END) AS deaths_2023
           FROM cdc_2023_county_deaths_under_75yrs
          GROUP BY (lpad(TRIM(BOTH FROM cdc_2023_county_deaths_under_75yrs.county_fips::text), 5, '0'::text))) cdc ON vcm.county_fips = cdc.county_fips
     LEFT JOIN ( SELECT lpad(TRIM(BOTH FROM "vw_2023_age_adj_YPLL".county_fips), 5, '0'::text) AS county_fips,
            "vw_2023_age_adj_YPLL".ct5_ypll_u75_age_adj_per_100k_low AS ct5_ypll_per_100k_low,
            "vw_2023_age_adj_YPLL".ct5_ypll_u75_age_adj_per_100k_mid AS ct5_ypll_per_100k_mid,
            "vw_2023_age_adj_YPLL".ct5_ypll_u75_age_adj_per_100k_high AS ct5_ypll_per_100k_high
           FROM "vw_2023_age_adj_YPLL") ypll2023 ON vcm.county_fips = ypll2023.county_fips
     LEFT JOIN ( SELECT lpad(TRIM(BOTH FROM vw_2023_hospital_deaths.county_fips), 5, '0'::text) AS county_fips,
            vw_2023_hospital_deaths.total_age_adjusted_deaths
           FROM vw_2023_hospital_deaths) hospdeaths2023 ON vcm.county_fips = hospdeaths2023.county_fips
  WHERE vcm.population IS NOT NULL AND vcm.population::numeric > 0::numeric AND vcm.county_fips !~~ '%000'::text
  ORDER BY vcm.county_fips;
