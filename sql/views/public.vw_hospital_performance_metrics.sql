CREATE OR REPLACE VIEW public."vw_hospital_performance_metrics" AS
 WITH nber_pivot AS (
         SELECT d.facility_id,
            max(d.facility_name::text) AS facility_name,
            max(d.address::text) AS address,
            max(d."city/town"::text) AS city_town,
            max(d.state::text) AS state,
            max(d.zip_code::text) AS zip_code,
            max(d."county/parish"::text) AS county_parish,
            max(d.telephone_number::text) AS telephone_number,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_90'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_90_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'COMP_HIP_KNEE'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS comp_hip_knee_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'MORT_30_AMI'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS mort_30_ami_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'MORT_30_CABG'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS mort_30_cabg_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'MORT_30_COPD'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS mort_30_copd_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'MORT_30_HF'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS mort_30_hf_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'MORT_30_PN'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS mort_30_pn_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'MORT_30_STK'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS mort_30_stk_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_03'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_03_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_04'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_04_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_06'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_06_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_08'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_08_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_09'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_09_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_10'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_10_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_11'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_11_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_12'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_12_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_13'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_13_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_14'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_14_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'PSI_15'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS psi_15_score
           FROM nber_complications_and_deaths d
          WHERE (d.measure_id::text = ANY (ARRAY['COMP_HIP_KNEE'::character varying, 'MORT_30_AMI'::character varying, 'MORT_30_CABG'::character varying, 'MORT_30_COPD'::character varying, 'MORT_30_HF'::character varying, 'MORT_30_PN'::character varying, 'MORT_30_STK'::character varying, 'PSI_03'::character varying, 'PSI_04'::character varying, 'PSI_06'::character varying, 'PSI_08'::character varying, 'PSI_09'::character varying, 'PSI_10'::character varying, 'PSI_11'::character varying, 'PSI_12'::character varying, 'PSI_13'::character varying, 'PSI_14'::character varying, 'PSI_15'::character varying, 'PSI_90'::character varying]::text[])) AND d.score::text <> 'Not Available'::text
          GROUP BY d.facility_id
        )
 SELECT s.id AS aha_id,
    s.bsc,
    s.robohos,
    s.robosys,
    s.roboven,
    s.adjpd,
    s.wfaipsn,
    s.wfaippd,
    s.wfaiss,
    s.wfaiart,
    s.wfaioacw,
    x.cms_facility_id,
    n.facility_id,
    n.facility_name,
    n.address,
    n.city_town,
    n.state,
    n.zip_code,
    n.county_parish,
    n.telephone_number,
    n.psi_90_score,
    n.comp_hip_knee_score,
    n.mort_30_ami_score,
    n.mort_30_cabg_score,
    n.mort_30_copd_score,
    n.mort_30_hf_score,
    n.mort_30_pn_score,
    n.mort_30_stk_score,
    n.psi_03_score,
    n.psi_04_score,
    n.psi_06_score,
    n.psi_08_score,
    n.psi_09_score,
    n.psi_10_score,
    n.psi_11_score,
    n.psi_12_score,
    n.psi_13_score,
    n.psi_14_score,
    n.psi_15_score
   FROM aha_survey_data s
     JOIN aha_cms_crosswalk x ON x.aha_id = s.id::text AND (x.match_status = ANY (ARRAY['AUTO_MATCH'::text, 'HUMAN_MATCH'::text]))
     LEFT JOIN nber_pivot n ON n.facility_id::text = x.cms_facility_id;
