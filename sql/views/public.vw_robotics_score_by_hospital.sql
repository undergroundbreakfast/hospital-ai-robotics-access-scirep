CREATE OR REPLACE VIEW public."vw_robotics_score_by_hospital" AS
 WITH hospital_details AS (
         SELECT aha_survey_data.id,
            aha_survey_data.mname,
            lpad(aha_survey_data.fcounty::text, 5, '0'::text) AS county_fips,
            COALESCE(aha_survey_data.bdtot, '0'::character varying)::integer AS bdtot,
            COALESCE(aha_survey_data.robohos, '0'::character varying)::integer * 10 + COALESCE(aha_survey_data.robosys, '0'::character varying)::integer * 3 + COALESCE(aha_survey_data.roboven, '0'::character varying)::integer * 2 AS robotics_adoption_score,
            TRIM(BOTH ', '::text FROM (
                CASE
                    WHEN COALESCE(aha_survey_data.robohos, '0'::character varying)::integer = 1 THEN 'robohos, '::text
                    ELSE ''::text
                END ||
                CASE
                    WHEN COALESCE(aha_survey_data.robosys, '0'::character varying)::integer = 1 THEN 'robosys, '::text
                    ELSE ''::text
                END) ||
                CASE
                    WHEN COALESCE(aha_survey_data.roboven, '0'::character varying)::integer = 1 THEN 'roboven, '::text
                    ELSE ''::text
                END) AS enabled_robotics_vars
           FROM aha_survey_data
        )
 SELECT id,
    mname,
    county_fips,
    bdtot,
    robotics_adoption_score,
    enabled_robotics_vars
   FROM hospital_details;
