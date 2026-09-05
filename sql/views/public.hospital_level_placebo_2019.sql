CREATE OR REPLACE VIEW public."hospital_level_placebo_2019" AS
 WITH comp AS (
         SELECT f_2019_nber_complications_and_deaths_hospital.facility_id,
            max(
                CASE
                    WHEN f_2019_nber_complications_and_deaths_hospital.measure_id::text = 'MORT_30_PN'::text THEN f_2019_nber_complications_and_deaths_hospital.score
                    ELSE NULL::character varying
                END::text) AS mort_30_pn_2019,
            max(
                CASE
                    WHEN f_2019_nber_complications_and_deaths_hospital.measure_id::text = 'MORT_30_HF'::text THEN f_2019_nber_complications_and_deaths_hospital.score
                    ELSE NULL::character varying
                END::text) AS mort_30_hf_2019,
            max(
                CASE
                    WHEN f_2019_nber_complications_and_deaths_hospital.measure_id::text = 'PSI_12_POSTOP_PULMEMB_DVT'::text THEN f_2019_nber_complications_and_deaths_hospital.score
                    ELSE NULL::character varying
                END::text) AS psi_12_2019,
            max(
                CASE
                    WHEN f_2019_nber_complications_and_deaths_hospital.measure_id::text = 'PSI_13_POST_SEPSIS'::text THEN f_2019_nber_complications_and_deaths_hospital.score
                    ELSE NULL::character varying
                END::text) AS psi_13_2019
           FROM f_2019_nber_complications_and_deaths_hospital
          WHERE f_2019_nber_complications_and_deaths_hospital.measure_id::text = ANY (ARRAY['MORT_30_PN'::character varying, 'MORT_30_HF'::character varying, 'PSI_12_POSTOP_PULMEMB_DVT'::character varying, 'PSI_13_POST_SEPSIS'::character varying]::text[])
          GROUP BY f_2019_nber_complications_and_deaths_hospital.facility_id
        ), tec AS (
         SELECT f_2019_nber_timely_and_effective_care_hospital.facility_id,
            max(
                CASE
                    WHEN f_2019_nber_timely_and_effective_care_hospital.measure_id::text = 'SEP_1'::text THEN f_2019_nber_timely_and_effective_care_hospital.score
                    ELSE NULL::character varying
                END::text) AS sep_1_2019,
            max(
                CASE
                    WHEN f_2019_nber_timely_and_effective_care_hospital.measure_id::text = 'OP_18b'::text THEN f_2019_nber_timely_and_effective_care_hospital.score
                    ELSE NULL::character varying
                END::text) AS op_18b_2019
           FROM f_2019_nber_timely_and_effective_care_hospital
          WHERE f_2019_nber_timely_and_effective_care_hospital.measure_id::text = ANY (ARRAY['SEP_1'::character varying, 'OP_18b'::character varying]::text[])
          GROUP BY f_2019_nber_timely_and_effective_care_hospital.facility_id
        ), upv AS (
         SELECT f_2019_nber_unplanned_hospital_visits_hospital.facility_id,
            max(
                CASE
                    WHEN f_2019_nber_unplanned_hospital_visits_hospital.measure_id::text = 'READM_30_PN'::text THEN f_2019_nber_unplanned_hospital_visits_hospital.score
                    ELSE NULL::character varying
                END::text) AS readm_30_pn_2019
           FROM f_2019_nber_unplanned_hospital_visits_hospital
          WHERE f_2019_nber_unplanned_hospital_visits_hospital.measure_id::text = 'READM_30_PN'::text
          GROUP BY f_2019_nber_unplanned_hospital_visits_hospital.facility_id
        )
 SELECT c.facility_id,
    c.mort_30_pn_2019,
    c.mort_30_hf_2019,
    c.psi_12_2019,
    c.psi_13_2019,
    t.sep_1_2019,
    t.op_18b_2019,
    u.readm_30_pn_2019
   FROM comp c
     LEFT JOIN tec t ON t.facility_id::text = c.facility_id::text
     LEFT JOIN upv u ON u.facility_id::text = c.facility_id::text
  ORDER BY c.facility_id;
