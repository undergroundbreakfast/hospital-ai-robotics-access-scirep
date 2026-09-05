CREATE OR REPLACE VIEW public."vw_timely_effective_care" AS
 WITH crosswalk AS (
         SELECT aha_cms_crosswalk.aha_id,
            aha_cms_crosswalk.cms_facility_id
           FROM aha_cms_crosswalk
          WHERE aha_cms_crosswalk.match_status = ANY (ARRAY['AUTO_MATCH'::text, 'HUMAN_MATCH'::text])
        ), effective AS (
         SELECT nber_timely_and_effective_care_hospital.facility_id,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'OP_29'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef1,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'OP_23'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef2,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'OP_18c'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef3,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'STK_02'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef4,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'HH_02'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef5,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'OP_18b'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef6,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'OP_22'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef7,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'SEV_SEP_6HR'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef8,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'STK_03'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef9,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'OP_40'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef10,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'ED_2_Strata_1'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef11,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'SAFE_USE_OF_OPIOIDS'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef12,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'STK_05'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef13,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'STK_06'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef14,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'SEV_SEP_3HR'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef15,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'IMM_3'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef16,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'SEP_SH_3HR'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef17,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'HCP_COVID_19'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef18,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'HH_01'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef19,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'OP_31'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef20,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'VTE_2'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef21,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'EDV'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::text
                    ELSE NULL::text
                END) AS ef22,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'SEP_1'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef23,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'SEP_SH_6HR'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef24,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'VTE_1'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef25,
            max(
                CASE
                    WHEN nber_timely_and_effective_care_hospital.measure_id::text = 'ED_2_Strata_2'::text AND (nber_timely_and_effective_care_hospital.score::text <> ALL (ARRAY['Not Available'::text, 'Not Applicable'::text])) THEN nber_timely_and_effective_care_hospital.score::numeric
                    ELSE NULL::numeric
                END) AS ef26
           FROM nber_timely_and_effective_care_hospital
          GROUP BY nber_timely_and_effective_care_hospital.facility_id
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
    ef.ef1,
    ef.ef2,
    ef.ef3,
    ef.ef4,
    ef.ef5,
    ef.ef6,
    ef.ef7,
    ef.ef8,
    ef.ef9,
    ef.ef10,
    ef.ef11,
    ef.ef12,
    ef.ef13,
    ef.ef14,
    ef.ef15,
    ef.ef16,
    ef.ef17,
    ef.ef18,
    ef.ef19,
    ef.ef20,
    ef.ef21,
    ef.ef22,
    ef.ef23,
    ef.ef24,
    ef.ef25,
    ef.ef26
   FROM crosswalk cw
     JOIN aha_survey_data aha ON aha.id::text = cw.aha_id
     LEFT JOIN effective ef ON ef.facility_id::text = cw.cms_facility_id;
