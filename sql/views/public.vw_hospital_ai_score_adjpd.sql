CREATE OR REPLACE VIEW public."vw_hospital_ai_score_adjpd" AS
 WITH hospital_details AS (
         SELECT aha_survey_data.id,
            aha_survey_data.mname,
            lpad(aha_survey_data.fcounty::text, 5, '0'::text) AS county_fips,
            COALESCE(aha_survey_data.bdtot, '0'::character varying)::integer AS bdtot,
            COALESCE(aha_survey_data.adjpd, '0'::character varying)::integer AS adjpd,
            COALESCE(aha_survey_data.wfaipsn, '0'::character varying)::integer + COALESCE(aha_survey_data.wfaippd, '0'::character varying)::integer + COALESCE(aha_survey_data.wfaiss, '0'::character varying)::integer + COALESCE(aha_survey_data.wfaiart, '0'::character varying)::integer + COALESCE(aha_survey_data.wfaioacw, '0'::character varying)::integer AS ai_adoption_score,
            TRIM(BOTH ', '::text FROM (((
                CASE
                    WHEN COALESCE(aha_survey_data.wfaipsn, '0'::character varying)::integer = 1 THEN 'wfaipsn, '::text
                    ELSE ''::text
                END ||
                CASE
                    WHEN COALESCE(aha_survey_data.wfaippd, '0'::character varying)::integer = 1 THEN 'wfaippd, '::text
                    ELSE ''::text
                END) ||
                CASE
                    WHEN COALESCE(aha_survey_data.wfaiss, '0'::character varying)::integer = 1 THEN 'wfaiss, '::text
                    ELSE ''::text
                END) ||
                CASE
                    WHEN COALESCE(aha_survey_data.wfaiart, '0'::character varying)::integer = 1 THEN 'wfaiart, '::text
                    ELSE ''::text
                END) ||
                CASE
                    WHEN COALESCE(aha_survey_data.wfaioacw, '0'::character varying)::integer = 1 THEN 'wfaioacw, '::text
                    ELSE ''::text
                END) AS enabled_ai_vars
           FROM aha_survey_data
        )
 SELECT id,
    mname,
    county_fips,
    bdtot,
    adjpd,
    ai_adoption_score,
    enabled_ai_vars
   FROM hospital_details;;
