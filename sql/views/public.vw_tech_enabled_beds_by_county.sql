CREATE OR REPLACE VIEW public."vw_tech_enabled_beds_by_county" AS
 WITH hospital_tech AS (
         SELECT lpad(aha_survey_data.fcounty::text, 5, '0'::text) AS county_fips,
            COALESCE(aha_survey_data.bdtot, '0'::character varying)::integer AS bdtot,
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
                END) AS tech_enabled_beds
           FROM hospital_tech
          GROUP BY hospital_tech.county_fips
        )
 SELECT cs.county_fips,
    round(cs.total_weighted_score::numeric / NULLIF(cs.total_beds, 0)::numeric, 5) AS weighted_avg_tech_score,
    cs.tech_enabled_beds,
    cs.total_beds,
    cs.tech_enabled_beds::numeric * 100.0 / NULLIF(cs.total_beds, 0)::numeric AS pct_tech_enabled_beds,
    m.deaths,
    m.population,
    m.crude_rate,
    h.irr_county_value
   FROM county_agg cs
     LEFT JOIN cdc_wonder_5_year_mortality_2023 m ON lpad(m.county_code::text, 5, '0'::text) = cs.county_fips
     LEFT JOIN hrsa_health_equity_data h ON lpad(h.county_fips_code::text, 5, '0'::text) = cs.county_fips;
