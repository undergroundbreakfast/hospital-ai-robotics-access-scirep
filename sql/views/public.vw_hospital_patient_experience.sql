CREATE OR REPLACE VIEW public."vw_hospital_patient_experience" AS
 WITH nber_hcahps_clean AS (
         SELECT nber_hcahps_hospital.facility_id,
            nber_hcahps_hospital.facility_name,
            nber_hcahps_hospital.address,
            nber_hcahps_hospital.city_town,
            nber_hcahps_hospital.state,
            nber_hcahps_hospital.zip_code,
            nber_hcahps_hospital.county_parish,
            nber_hcahps_hospital.hcahps_measure_id,
                CASE
                    WHEN nber_hcahps_hospital.hcahps_answer_percent::text = ANY (ARRAY['Not Applicable'::character varying, 'Not Available'::character varying]::text[]) THEN NULL::character varying
                    ELSE nber_hcahps_hospital.hcahps_answer_percent
                END AS hcahps_answer_percent,
                CASE
                    WHEN nber_hcahps_hospital.patient_survey_star_rating::text = ANY (ARRAY['Not Applicable'::character varying, 'Not Available'::character varying]::text[]) THEN NULL::character varying
                    ELSE nber_hcahps_hospital.patient_survey_star_rating
                END AS patient_survey_star_rating,
                CASE
                    WHEN nber_hcahps_hospital.survey_response_rate_percent::text = ANY (ARRAY['Not Applicable'::character varying, 'Not Available'::character varying]::text[]) THEN NULL::character varying
                    ELSE nber_hcahps_hospital.survey_response_rate_percent
                END AS survey_response_rate_percent
           FROM nber_hcahps_hospital
        ), nber_hcahps_pivot AS (
         SELECT h.facility_id,
            max(h.facility_name::text) AS facility_name,
            max(h.address::text) AS address,
            max(h.city_town::text) AS city_town,
            max(h.state::text) AS state,
            max(h.zip_code::text) AS zip_code,
            max(h.county_parish::text) AS county_parish,
            max(h.patient_survey_star_rating::text) AS patient_survey_star_rating,
            max(h.survey_response_rate_percent::text) AS survey_response_rate_percent,
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_BATH_HELP_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_BATH_HELP_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_BATH_HELP_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_BATH_HELP_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_BATH_HELP_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_BATH_HELP_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CALL_BUTTON_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CALL_BUTTON_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CALL_BUTTON_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CALL_BUTTON_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CALL_BUTTON_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CALL_BUTTON_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CLEAN_HSP_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CLEAN_HSP_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CLEAN_HSP_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CLEAN_HSP_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CLEAN_HSP_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CLEAN_HSP_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CLEAN_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CLEAN_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CLEAN_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CLEAN_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_1_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_1_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_1_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_1_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_1_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_1_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_1_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_1_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_1_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_1_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_2_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_2_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_2_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_2_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_2_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_2_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_2_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_2_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_2_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_2_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_3_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_3_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_3_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_3_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_3_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_3_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_3_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_3_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_3_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_3_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_5_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_5_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_5_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_5_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_5_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_5_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_5_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_5_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_5_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_5_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_6_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_6_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_6_N_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_6_N_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_6_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_6_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_6_Y_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_6_Y_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_7_A'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_7_A",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_7_D_SD'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_7_D_SD",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_7_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_7_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_7_SA'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_7_SA",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_COMP_7_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_COMP_7_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CT_MED_A'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CT_MED_A",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CT_MED_D_SD'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CT_MED_D_SD",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CT_MED_SA'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CT_MED_SA",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CT_PREFER_A'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CT_PREFER_A",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CT_PREFER_D_SD'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CT_PREFER_D_SD",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CT_PREFER_SA'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CT_PREFER_SA",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CT_UNDER_A'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CT_UNDER_A",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CT_UNDER_D_SD'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CT_UNDER_D_SD",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_CT_UNDER_SA'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_CT_UNDER_SA",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DISCH_HELP_N_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DISCH_HELP_N_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DISCH_HELP_Y_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DISCH_HELP_Y_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DOCTOR_EXPLAIN_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DOCTOR_EXPLAIN_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DOCTOR_EXPLAIN_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DOCTOR_EXPLAIN_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DOCTOR_EXPLAIN_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DOCTOR_EXPLAIN_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DOCTOR_LISTEN_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DOCTOR_LISTEN_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DOCTOR_LISTEN_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DOCTOR_LISTEN_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DOCTOR_LISTEN_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DOCTOR_LISTEN_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DOCTOR_RESPECT_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DOCTOR_RESPECT_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DOCTOR_RESPECT_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DOCTOR_RESPECT_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_DOCTOR_RESPECT_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_DOCTOR_RESPECT_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_HSP_RATING_0_6'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_HSP_RATING_0_6",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_HSP_RATING_7_8'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_HSP_RATING_7_8",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_HSP_RATING_9_10'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_HSP_RATING_9_10",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_HSP_RATING_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_HSP_RATING_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_HSP_RATING_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_HSP_RATING_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_MED_FOR_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_MED_FOR_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_MED_FOR_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_MED_FOR_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_MED_FOR_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_MED_FOR_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_NURSE_EXPLAIN_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_NURSE_EXPLAIN_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_NURSE_EXPLAIN_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_NURSE_EXPLAIN_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_NURSE_EXPLAIN_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_NURSE_EXPLAIN_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_NURSE_LISTEN_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_NURSE_LISTEN_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_NURSE_LISTEN_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_NURSE_LISTEN_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_NURSE_LISTEN_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_NURSE_LISTEN_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_NURSE_RESPECT_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_NURSE_RESPECT_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_NURSE_RESPECT_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_NURSE_RESPECT_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_NURSE_RESPECT_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_NURSE_RESPECT_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_QUIET_HSP_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_QUIET_HSP_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_QUIET_HSP_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_QUIET_HSP_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_QUIET_HSP_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_QUIET_HSP_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_QUIET_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_QUIET_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_QUIET_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_QUIET_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_RECMND_DN'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_RECMND_DN",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_RECMND_DY'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_RECMND_DY",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_RECMND_LINEAR_SCORE'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_RECMND_LINEAR_SCORE",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_RECMND_PY'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_RECMND_PY",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_RECMND_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_RECMND_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_SIDE_EFFECTS_A_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_SIDE_EFFECTS_A_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_SIDE_EFFECTS_SN_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_SIDE_EFFECTS_SN_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_SIDE_EFFECTS_U_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_SIDE_EFFECTS_U_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_STAR_RATING'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_STAR_RATING",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_SYMPTOMS_N_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_SYMPTOMS_N_P",
            max(
                CASE
                    WHEN h.hcahps_measure_id::text = 'H_SYMPTOMS_Y_P'::text THEN h.hcahps_answer_percent
                    ELSE NULL::character varying
                END::text) AS "H_SYMPTOMS_Y_P"
           FROM nber_hcahps_clean h
          GROUP BY h.facility_id
        )
 SELECT a.id AS aha_id,
    a.bsc,
    a.sysname,
    a.robohos,
    a.robosys,
    a.roboven,
    a.adjpd,
    a.wfaipsn,
    a.wfaippd,
    a.wfaiss,
    a.wfaiart,
    a.wfaioacw,
    p.facility_id,
    p.facility_name,
    p.address,
    p.city_town,
    p.state,
    p.zip_code,
    p.county_parish,
    p.patient_survey_star_rating,
    p.survey_response_rate_percent,
    p."H_BATH_HELP_A_P",
    p."H_BATH_HELP_SN_P",
    p."H_BATH_HELP_U_P",
    p."H_CALL_BUTTON_A_P",
    p."H_CALL_BUTTON_SN_P",
    p."H_CALL_BUTTON_U_P",
    p."H_CLEAN_HSP_A_P",
    p."H_CLEAN_HSP_SN_P",
    p."H_CLEAN_HSP_U_P",
    p."H_CLEAN_LINEAR_SCORE",
    p."H_CLEAN_STAR_RATING",
    p."H_COMP_1_A_P",
    p."H_COMP_1_LINEAR_SCORE",
    p."H_COMP_1_SN_P",
    p."H_COMP_1_STAR_RATING",
    p."H_COMP_1_U_P",
    p."H_COMP_2_A_P",
    p."H_COMP_2_LINEAR_SCORE",
    p."H_COMP_2_SN_P",
    p."H_COMP_2_STAR_RATING",
    p."H_COMP_2_U_P",
    p."H_COMP_3_A_P",
    p."H_COMP_3_LINEAR_SCORE",
    p."H_COMP_3_SN_P",
    p."H_COMP_3_STAR_RATING",
    p."H_COMP_3_U_P",
    p."H_COMP_5_A_P",
    p."H_COMP_5_LINEAR_SCORE",
    p."H_COMP_5_SN_P",
    p."H_COMP_5_STAR_RATING",
    p."H_COMP_5_U_P",
    p."H_COMP_6_LINEAR_SCORE",
    p."H_COMP_6_N_P",
    p."H_COMP_6_STAR_RATING",
    p."H_COMP_6_Y_P",
    p."H_COMP_7_A",
    p."H_COMP_7_D_SD",
    p."H_COMP_7_LINEAR_SCORE",
    p."H_COMP_7_SA",
    p."H_COMP_7_STAR_RATING",
    p."H_CT_MED_A",
    p."H_CT_MED_D_SD",
    p."H_CT_MED_SA",
    p."H_CT_PREFER_A",
    p."H_CT_PREFER_D_SD",
    p."H_CT_PREFER_SA",
    p."H_CT_UNDER_A",
    p."H_CT_UNDER_D_SD",
    p."H_CT_UNDER_SA",
    p."H_DISCH_HELP_N_P",
    p."H_DISCH_HELP_Y_P",
    p."H_DOCTOR_EXPLAIN_A_P",
    p."H_DOCTOR_EXPLAIN_SN_P",
    p."H_DOCTOR_EXPLAIN_U_P",
    p."H_DOCTOR_LISTEN_A_P",
    p."H_DOCTOR_LISTEN_SN_P",
    p."H_DOCTOR_LISTEN_U_P",
    p."H_DOCTOR_RESPECT_A_P",
    p."H_DOCTOR_RESPECT_SN_P",
    p."H_DOCTOR_RESPECT_U_P",
    p."H_HSP_RATING_0_6",
    p."H_HSP_RATING_7_8",
    p."H_HSP_RATING_9_10",
    p."H_HSP_RATING_LINEAR_SCORE",
    p."H_HSP_RATING_STAR_RATING",
    p."H_MED_FOR_A_P",
    p."H_MED_FOR_SN_P",
    p."H_MED_FOR_U_P",
    p."H_NURSE_EXPLAIN_A_P",
    p."H_NURSE_EXPLAIN_SN_P",
    p."H_NURSE_EXPLAIN_U_P",
    p."H_NURSE_LISTEN_A_P",
    p."H_NURSE_LISTEN_SN_P",
    p."H_NURSE_LISTEN_U_P",
    p."H_NURSE_RESPECT_A_P",
    p."H_NURSE_RESPECT_SN_P",
    p."H_NURSE_RESPECT_U_P",
    p."H_QUIET_HSP_A_P",
    p."H_QUIET_HSP_SN_P",
    p."H_QUIET_HSP_U_P",
    p."H_QUIET_LINEAR_SCORE",
    p."H_QUIET_STAR_RATING",
    p."H_RECMND_DN",
    p."H_RECMND_DY",
    p."H_RECMND_LINEAR_SCORE",
    p."H_RECMND_PY",
    p."H_RECMND_STAR_RATING",
    p."H_SIDE_EFFECTS_A_P",
    p."H_SIDE_EFFECTS_SN_P",
    p."H_SIDE_EFFECTS_U_P",
    p."H_STAR_RATING",
    p."H_SYMPTOMS_N_P",
    p."H_SYMPTOMS_Y_P"
   FROM aha_survey_data a
     JOIN aha_cms_crosswalk c ON c.aha_id = a.id::text AND (c.match_status = ANY (ARRAY['AUTO_MATCH'::text, 'HUMAN_MATCH'::text]))
     LEFT JOIN nber_hcahps_pivot p ON p.facility_id::text = c.cms_facility_id;
