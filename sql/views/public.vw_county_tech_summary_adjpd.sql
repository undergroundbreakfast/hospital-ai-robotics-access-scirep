CREATE OR REPLACE VIEW public.vw_county_tech_summary_adjpd AS
 WITH hospital_tech AS (
         SELECT lpad(aha_survey_data.fcounty::text, 5, '0'::text) AS county_fips,
            COALESCE(aha_survey_data.adjpd, '0'::character varying)::integer AS adjpd,
            COALESCE(aha_survey_data.robohos, '0'::character varying)::integer AS robohos,
            COALESCE(aha_survey_data.robosys, '0'::character varying)::integer AS robosys,
            COALESCE(aha_survey_data.roboven, '0'::character varying)::integer AS roboven,
            COALESCE(aha_survey_data.wfaipsn, '0'::character varying)::integer AS wfaipsn,
            COALESCE(aha_survey_data.wfaippd, '0'::character varying)::integer AS wfaippd,
            COALESCE(aha_survey_data.wfaiss, '0'::character varying)::integer AS wfaiss,
            COALESCE(aha_survey_data.wfaiart, '0'::character varying)::integer AS wfaiart,
            COALESCE(aha_survey_data.wfaioacw, '0'::character varying)::integer AS wfaioacw,
            COALESCE(aha_survey_data.wfaidna, '0'::character varying)::integer AS wfaidna,
            COALESCE(aha_survey_data.robohos, '0'::character varying)::integer + COALESCE(aha_survey_data.robosys, '0'::character varying)::integer + COALESCE(aha_survey_data.roboven, '0'::character varying)::integer + COALESCE(aha_survey_data.wfaipsn, '0'::character varying)::integer + COALESCE(aha_survey_data.wfaippd, '0'::character varying)::integer + COALESCE(aha_survey_data.wfaiss, '0'::character varying)::integer + COALESCE(aha_survey_data.wfaiart, '0'::character varying)::integer + COALESCE(aha_survey_data.wfaioacw, '0'::character varying)::integer AS tech_score
           FROM aha_survey_data
        ), county_agg AS (
         SELECT hospital_tech.county_fips,
            sum(hospital_tech.tech_score * hospital_tech.adjpd) AS total_weighted_score,
            sum(hospital_tech.adjpd) AS total_patient_days,
            sum(
                CASE
                    WHEN hospital_tech.tech_score > 0 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS tech_enabled_days,
            sum(
                CASE
                    WHEN hospital_tech.robohos = 1 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS robohos_enabled_days,
            sum(
                CASE
                    WHEN hospital_tech.robosys = 1 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS robosys_enabled_days,
            sum(
                CASE
                    WHEN hospital_tech.roboven = 1 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS roboven_enabled_days,
            sum(
                CASE
                    WHEN hospital_tech.wfaipsn = 1 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS wfaipsn_enabled_days,
            sum(
                CASE
                    WHEN hospital_tech.wfaippd = 1 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS wfaippd_enabled_days,
            sum(
                CASE
                    WHEN hospital_tech.wfaiss = 1 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS wfaiss_enabled_days,
            sum(
                CASE
                    WHEN hospital_tech.wfaiart = 1 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS wfaiart_enabled_days,
            sum(
                CASE
                    WHEN hospital_tech.wfaioacw = 1 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS wfaioacw_enabled_days,
            sum(
                CASE
                    WHEN hospital_tech.wfaidna = 1 THEN hospital_tech.adjpd
                    ELSE 0
                END) AS wfaidna_enabled_days
           FROM hospital_tech
          GROUP BY hospital_tech.county_fips
        ), all_counties AS (
         SELECT lpad(hrsa_health_equity_data.county_fips_code::text, 5, '0'::text) AS county_fips,
            hrsa_health_equity_data.irr_county_value
           FROM hrsa_health_equity_data
          WHERE "right"(lpad(hrsa_health_equity_data.county_fips_code::text, 5, '0'::text), 3) <> '000'::text
        )
 SELECT ac.county_fips,
    m.deaths,
    m.population,
    m.crude_rate,
    ac.irr_county_value,
    COALESCE(ca.total_patient_days, 0::bigint) AS total_patient_days,
    COALESCE(ca.tech_enabled_days, 0::bigint) AS tech_enabled_days,
    COALESCE(ca.total_weighted_score, 0::bigint) AS total_weighted_score,
    round(COALESCE(ca.total_weighted_score, 0::bigint)::numeric / NULLIF(COALESCE(ca.total_patient_days, 0::bigint)::numeric, 0::numeric), 5) AS weighted_avg_tech_score,
    COALESCE(ca.robohos_enabled_days, 0::bigint) AS robohos_enabled_days,
    COALESCE(ca.robosys_enabled_days, 0::bigint) AS robosys_enabled_days,
    COALESCE(ca.roboven_enabled_days, 0::bigint) AS roboven_enabled_days,
    COALESCE(ca.wfaipsn_enabled_days, 0::bigint) AS wfaipsn_enabled_days,
    COALESCE(ca.wfaippd_enabled_days, 0::bigint) AS wfaippd_enabled_days,
    COALESCE(ca.wfaiss_enabled_days, 0::bigint) AS wfaiss_enabled_days,
    COALESCE(ca.wfaiart_enabled_days, 0::bigint) AS wfaiart_enabled_days,
    COALESCE(ca.wfaioacw_enabled_days, 0::bigint) AS wfaioacw_enabled_days,
    COALESCE(ca.wfaidna_enabled_days, 0::bigint) AS wfaidna_enabled_days,
    COALESCE(ca.robohos_enabled_days::numeric * 100::numeric / NULLIF(ca.total_patient_days::numeric, 0::numeric), 0::numeric) AS pct_robohos_enabled,
    COALESCE(ca.wfaiss_enabled_days::numeric * 100::numeric / NULLIF(ca.total_patient_days::numeric, 0::numeric), 0::numeric) AS pct_wfaiss_enabled,
    COALESCE(ca.wfaipsn_enabled_days::numeric * 100::numeric / NULLIF(ca.total_patient_days::numeric, 0::numeric), 0::numeric) AS pct_wfaipsn_enabled,
    COALESCE(ca.wfaippd_enabled_days::numeric * 100::numeric / NULLIF(ca.total_patient_days::numeric, 0::numeric), 0::numeric) AS pct_wfaippd_enabled,
    COALESCE(ca.wfaiart_enabled_days::numeric * 100::numeric / NULLIF(ca.total_patient_days::numeric, 0::numeric), 0::numeric) AS pct_wfaiart_enabled,
    COALESCE(ca.wfaidna_enabled_days::numeric * 100::numeric / NULLIF(ca.total_patient_days::numeric, 0::numeric), 0::numeric) AS pct_wfaidna_enabled,
    COALESCE(ca.wfaioacw_enabled_days::numeric * 100::numeric / NULLIF(ca.total_patient_days::numeric, 0::numeric), 0::numeric) AS pct_wfaioacw_enabled
   FROM all_counties ac
     LEFT JOIN county_agg ca ON ca.county_fips = ac.county_fips
     LEFT JOIN cdc_wonder_5_year_mortality_2023 m ON lpad(m.county_code::text, 5, '0'::text) = ac.county_fips;;
