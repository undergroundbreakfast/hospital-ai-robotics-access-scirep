CREATE OR REPLACE VIEW public.vw_adjpd_weighted_tech_summary AS
 WITH hospital_tech AS (
         SELECT lpad(aha_survey_data.fcounty::text, 5, '0'::text) AS county_fips,
            COALESCE(aha_survey_data.bdtot, '0'::character varying)::integer AS bdtot,
            COALESCE(aha_survey_data.adjpd, '0'::character varying)::bigint AS adjpd,
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
            sum(hospital_tech.tech_score * hospital_tech.bdtot) AS total_weighted_score,
            sum(hospital_tech.bdtot) AS total_beds,
            sum(
                CASE
                    WHEN hospital_tech.tech_score > 0 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS tech_enabled_beds,
            sum(
                CASE
                    WHEN hospital_tech.robohos = 1 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS robohos_enabled_beds,
            sum(
                CASE
                    WHEN hospital_tech.robosys = 1 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS robosys_enabled_beds,
            sum(
                CASE
                    WHEN hospital_tech.roboven = 1 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS roboven_enabled_beds,
            sum(
                CASE
                    WHEN hospital_tech.wfaipsn = 1 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS wfaipsn_enabled_beds,
            sum(
                CASE
                    WHEN hospital_tech.wfaippd = 1 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS wfaippd_enabled_beds,
            sum(
                CASE
                    WHEN hospital_tech.wfaiss = 1 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS wfaiss_enabled_beds,
            sum(
                CASE
                    WHEN hospital_tech.wfaiart = 1 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS wfaiart_enabled_beds,
            sum(
                CASE
                    WHEN hospital_tech.wfaioacw = 1 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS wfaioacw_enabled_beds,
            sum(
                CASE
                    WHEN hospital_tech.wfaidna = 1 THEN hospital_tech.bdtot
                    ELSE 0
                END) AS wfaidna_enabled_beds,
            sum(hospital_tech.tech_score * hospital_tech.adjpd) AS total_weighted_score_adjpd,
            sum(hospital_tech.adjpd) AS total_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.tech_score > 0 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS tech_enabled_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.robohos = 1 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS robohos_enabled_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.robosys = 1 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS robosys_enabled_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.roboven = 1 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS roboven_enabled_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.wfaipsn = 1 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS wfaipsn_enabled_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.wfaippd = 1 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS wfaippd_enabled_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.wfaiss = 1 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS wfaiss_enabled_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.wfaiart = 1 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS wfaiart_enabled_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.wfaioacw = 1 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS wfaioacw_enabled_adjpd,
            sum(
                CASE
                    WHEN hospital_tech.wfaidna = 1 THEN hospital_tech.adjpd
                    ELSE 0::bigint
                END) AS wfaidna_enabled_adjpd
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
    COALESCE(ca.total_beds, 0::bigint) AS total_beds,
    COALESCE(ca.tech_enabled_beds, 0::bigint) AS tech_enabled_beds,
    COALESCE(ca.total_weighted_score, 0::bigint) AS total_weighted_score,
    round(ca.total_weighted_score::numeric / NULLIF(ca.total_beds, 0)::numeric, 5) AS weighted_avg_tech_score,
    COALESCE(ca.robohos_enabled_beds, 0::bigint) AS robohos_enabled_beds,
    COALESCE(ca.robosys_enabled_beds, 0::bigint) AS robosys_enabled_beds,
    COALESCE(ca.roboven_enabled_beds, 0::bigint) AS roboven_enabled_beds,
    COALESCE(ca.wfaipsn_enabled_beds, 0::bigint) AS wfaipsn_enabled_beds,
    COALESCE(ca.wfaippd_enabled_beds, 0::bigint) AS wfaippd_enabled_beds,
    COALESCE(ca.wfaiss_enabled_beds, 0::bigint) AS wfaiss_enabled_beds,
    COALESCE(ca.wfaiart_enabled_beds, 0::bigint) AS wfaiart_enabled_beds,
    COALESCE(ca.wfaioacw_enabled_beds, 0::bigint) AS wfaioacw_enabled_beds,
    COALESCE(ca.wfaidna_enabled_beds, 0::bigint) AS wfaidna_enabled_beds,
    COALESCE(ca.robohos_enabled_beds::numeric * 100::numeric / NULLIF(ca.total_beds, 0)::numeric, 0::numeric) AS pct_robohos_enabled_beds,
    COALESCE(ca.wfaiss_enabled_beds::numeric * 100::numeric / NULLIF(ca.total_beds, 0)::numeric, 0::numeric) AS pct_wfaiss_enabled_beds,
    COALESCE(ca.wfaipsn_enabled_beds::numeric * 100::numeric / NULLIF(ca.total_beds, 0)::numeric, 0::numeric) AS pct_wfaipsn_enabled_beds,
    COALESCE(ca.wfaippd_enabled_beds::numeric * 100::numeric / NULLIF(ca.total_beds, 0)::numeric, 0::numeric) AS pct_wfaippd_enabled_beds,
    COALESCE(ca.wfaiart_enabled_beds::numeric * 100::numeric / NULLIF(ca.total_beds, 0)::numeric, 0::numeric) AS pct_wfaiart_enabled_beds,
    COALESCE(ca.wfaidna_enabled_beds::numeric * 100::numeric / NULLIF(ca.total_beds, 0)::numeric, 0::numeric) AS pct_wfaidna_enabled_beds,
    COALESCE(ca.wfaioacw_enabled_beds::numeric * 100::numeric / NULLIF(ca.total_beds, 0)::numeric, 0::numeric) AS pct_wfaioacw_enabled_beds,
    COALESCE(ca.total_adjpd, 0::numeric) AS total_adjpd,
    COALESCE(ca.tech_enabled_adjpd, 0::numeric) AS tech_enabled_adjpd,
    COALESCE(ca.total_weighted_score_adjpd, 0::numeric) AS total_weighted_score_adjpd,
    round(ca.total_weighted_score_adjpd / NULLIF(ca.total_adjpd, 0::numeric), 5) AS weighted_avg_tech_score_adjpd,
    COALESCE(ca.robohos_enabled_adjpd, 0::numeric) AS robohos_enabled_adjpd,
    COALESCE(ca.robosys_enabled_adjpd, 0::numeric) AS robosys_enabled_adjpd,
    COALESCE(ca.roboven_enabled_adjpd, 0::numeric) AS roboven_enabled_adjpd,
    COALESCE(ca.wfaipsn_enabled_adjpd, 0::numeric) AS wfaipsn_enabled_adjpd,
    COALESCE(ca.wfaippd_enabled_adjpd, 0::numeric) AS wfaippd_enabled_adjpd,
    COALESCE(ca.wfaiss_enabled_adjpd, 0::numeric) AS wfaiss_enabled_adjpd,
    COALESCE(ca.wfaiart_enabled_adjpd, 0::numeric) AS wfaiart_enabled_adjpd,
    COALESCE(ca.wfaioacw_enabled_adjpd, 0::numeric) AS wfaioacw_enabled_adjpd,
    COALESCE(ca.wfaidna_enabled_adjpd, 0::numeric) AS wfaidna_enabled_adjpd,
    COALESCE(ca.robohos_enabled_adjpd * 100::numeric / NULLIF(ca.total_adjpd, 0::numeric), 0::numeric) AS pct_robohos_enabled_adjpd,
    COALESCE(ca.wfaiss_enabled_adjpd * 100::numeric / NULLIF(ca.total_adjpd, 0::numeric), 0::numeric) AS pct_wfaiss_enabled_adjpd,
    COALESCE(ca.wfaipsn_enabled_adjpd * 100::numeric / NULLIF(ca.total_adjpd, 0::numeric), 0::numeric) AS pct_wfaipsn_enabled_adjpd,
    COALESCE(ca.wfaippd_enabled_adjpd * 100::numeric / NULLIF(ca.total_adjpd, 0::numeric), 0::numeric) AS pct_wfaippd_enabled_adjpd,
    COALESCE(ca.wfaiart_enabled_adjpd * 100::numeric / NULLIF(ca.total_adjpd, 0::numeric), 0::numeric) AS pct_wfaiart_enabled_adjpd,
    COALESCE(ca.wfaidna_enabled_adjpd * 100::numeric / NULLIF(ca.total_adjpd, 0::numeric), 0::numeric) AS pct_wfaidna_enabled_adjpd,
    COALESCE(ca.wfaioacw_enabled_adjpd * 100::numeric / NULLIF(ca.total_adjpd, 0::numeric), 0::numeric) AS pct_wfaioacw_enabled_adjpd
   FROM all_counties ac
     LEFT JOIN county_agg ca ON ca.county_fips = ac.county_fips
     LEFT JOIN cdc_wonder_5_year_mortality_2023 m ON lpad(m.county_code::text, 5, '0'::text) = ac.county_fips;;
