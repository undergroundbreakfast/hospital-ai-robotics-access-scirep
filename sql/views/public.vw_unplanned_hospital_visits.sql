CREATE OR REPLACE VIEW public."vw_unplanned_hospital_visits" AS
 WITH crosswalk AS (
         SELECT aha_cms_crosswalk.aha_id,
            aha_cms_crosswalk.cms_facility_id
           FROM aha_cms_crosswalk
          WHERE aha_cms_crosswalk.match_status = ANY (ARRAY['AUTO_MATCH'::text, 'HUMAN_MATCH'::text])
        ), unplanned AS (
         SELECT nber_unplanned_hospital_visits_hospital.facility_id,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'READM_30_CABG'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS readm_30_cabg,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'READM_30_HF'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS readm_30_hf,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'OP_35_ED'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS op_35_ed,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'OP_36'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS op_36,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'OP_32'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS op_32,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'EDAC_30_AMI'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS edac_30_ami,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'OP_35_ADM'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS op_35_adm,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'READM_30_COPD'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS readm_30_copd,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'READM_30_PN'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS readm_30_pn,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'READM_30_AMI'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS readm_30_ami,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'READM_30_HIP_KNEE'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS readm_30_hip_knee,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'READM_30_HOSP_WIDE'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS readm_30_hosp_wide,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'EDAC_30_PN'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS edac_30_pn,
            max(
                CASE
                    WHEN nber_unplanned_hospital_visits_hospital.measure_id::text = 'EDAC_30_HF'::text AND (nber_unplanned_hospital_visits_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_unplanned_hospital_visits_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS edac_30_hf
           FROM nber_unplanned_hospital_visits_hospital
          GROUP BY nber_unplanned_hospital_visits_hospital.facility_id
        )
 SELECT aha.id AS aha_id,
    cw.cms_facility_id,
    aha.bsc,
    aha.sysname,
    aha.robohos,
    aha.robosys,
    aha.roboven,
    aha.adjpd,
    aha.wfaipsn,
    aha.wfaippd,
    aha.wfaiss,
    aha.wfaiart,
    aha.wfaioacw,
    up.readm_30_cabg,
    up.readm_30_hf,
    up.op_35_ed,
    up.op_36,
    up.op_32,
    up.edac_30_ami,
    up.op_35_adm,
    up.readm_30_copd,
    up.readm_30_pn,
    up.readm_30_ami,
    up.readm_30_hip_knee,
    up.readm_30_hosp_wide,
    up.edac_30_pn,
    up.edac_30_hf
   FROM aha_survey_data aha
     JOIN crosswalk cw ON cw.aha_id = aha.id::text
     LEFT JOIN unplanned up ON up.facility_id::text = cw.cms_facility_id;
