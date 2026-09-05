CREATE OR REPLACE VIEW public."vw_healthcare_associated_infections" AS
 WITH nber_hai_pivot AS (
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
                    WHEN d.measure_id::text = 'HAI_3_CILOWER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_3_cilower_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_6_ELIGCASES'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_6_eligcases_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_2_ELIGCASES'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_2_eligcases_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_1_ELIGCASES'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_1_eligcases_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_4_CIUPPER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_4_ciupper_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_1_NUMERATOR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_1_numerator_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_5_DOPC'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_5_dopc_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_6_DOPC'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_6_dopc_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_6_CILOWER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_6_cilower_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_5_SIR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_5_sir_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_4_ELIGCASES'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_4_eligcases_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_5_ELIGCASES'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_5_eligcases_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_4_NUMERATOR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_4_numerator_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_6_SIR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_6_sir_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_5_CIUPPER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_5_ciupper_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_2_NUMERATOR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_2_numerator_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_6_CIUPPER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_6_ciupper_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_4_DOPC'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_4_dopc_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_5_NUMERATOR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_5_numerator_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_1_SIR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_1_sir_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_4_CILOWER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_4_cilower_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_3_SIR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_3_sir_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_2_DOPC'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_2_dopc_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_4_SIR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_4_sir_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_6_NUMERATOR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_6_numerator_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_2_CIUPPER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_2_ciupper_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_3_CIUPPER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_3_ciupper_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_2_SIR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_2_sir_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_1_CILOWER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_1_cilower_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_2_CILOWER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_2_cilower_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_3_ELIGCASES'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_3_eligcases_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_3_NUMERATOR'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_3_numerator_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_5_CILOWER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_5_cilower_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_1_DOPC'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_1_dopc_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_3_DOPC'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_3_dopc_score,
            max(
                CASE
                    WHEN d.measure_id::text = 'HAI_1_CIUPPER'::text THEN d.score
                    ELSE NULL::character varying
                END::text) AS hai_1_ciupper_score
           FROM nber_healthcare_associated_infections d
          WHERE (d.measure_id::text = ANY (ARRAY['HAI_3_CILOWER'::character varying::text, 'HAI_6_ELIGCASES'::character varying::text, 'HAI_2_ELIGCASES'::character varying::text, 'HAI_1_ELIGCASES'::character varying::text, 'HAI_4_CIUPPER'::character varying::text, 'HAI_1_NUMERATOR'::character varying::text, 'HAI_5_DOPC'::character varying::text, 'HAI_6_DOPC'::character varying::text, 'HAI_6_CILOWER'::character varying::text, 'HAI_5_SIR'::character varying::text, 'HAI_4_ELIGCASES'::character varying::text, 'HAI_5_ELIGCASES'::character varying::text, 'HAI_4_NUMERATOR'::character varying::text, 'HAI_6_SIR'::character varying::text, 'HAI_5_CIUPPER'::character varying::text, 'HAI_2_NUMERATOR'::character varying::text, 'HAI_6_CIUPPER'::character varying::text, 'HAI_4_DOPC'::character varying::text, 'HAI_5_NUMERATOR'::character varying::text, 'HAI_1_SIR'::character varying::text, 'HAI_4_CILOWER'::character varying::text, 'HAI_3_SIR'::character varying::text, 'HAI_2_DOPC'::character varying::text, 'HAI_4_SIR'::character varying::text, 'HAI_6_NUMERATOR'::character varying::text, 'HAI_2_CIUPPER'::character varying::text, 'HAI_3_CIUPPER'::character varying::text, 'HAI_2_SIR'::character varying::text, 'HAI_1_CILOWER'::character varying::text, 'HAI_2_CILOWER'::character varying::text, 'HAI_3_ELIGCASES'::character varying::text, 'HAI_3_NUMERATOR'::character varying::text, 'HAI_5_CILOWER'::character varying::text, 'HAI_1_DOPC'::character varying::text, 'HAI_3_DOPC'::character varying::text, 'HAI_1_CIUPPER'::character varying::text])) AND d.score::text <> 'Not Available'::text
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
    h.facility_id,
    h.facility_name,
    h.address,
    h.city_town,
    h.state,
    h.zip_code,
    h.county_parish,
    h.telephone_number,
    h.hai_3_cilower_score,
    h.hai_6_eligcases_score,
    h.hai_2_eligcases_score,
    h.hai_1_eligcases_score,
    h.hai_4_ciupper_score,
    h.hai_1_numerator_score,
    h.hai_5_dopc_score,
    h.hai_6_dopc_score,
    h.hai_6_cilower_score,
    h.hai_5_sir_score,
    h.hai_4_eligcases_score,
    h.hai_5_eligcases_score,
    h.hai_4_numerator_score,
    h.hai_6_sir_score,
    h.hai_5_ciupper_score,
    h.hai_2_numerator_score,
    h.hai_6_ciupper_score,
    h.hai_4_dopc_score,
    h.hai_5_numerator_score,
    h.hai_1_sir_score,
    h.hai_4_cilower_score,
    h.hai_3_sir_score,
    h.hai_2_dopc_score,
    h.hai_4_sir_score,
    h.hai_6_numerator_score,
    h.hai_2_ciupper_score,
    h.hai_3_ciupper_score,
    h.hai_2_sir_score,
    h.hai_1_cilower_score,
    h.hai_2_cilower_score,
    h.hai_3_eligcases_score,
    h.hai_3_numerator_score,
    h.hai_5_cilower_score,
    h.hai_1_dopc_score,
    h.hai_3_dopc_score,
    h.hai_1_ciupper_score
   FROM aha_survey_data s
     JOIN aha_cms_crosswalk x ON x.aha_id = s.id::text AND (x.match_status = ANY (ARRAY['AUTO_MATCH'::text, 'HUMAN_MATCH'::text]))
     LEFT JOIN nber_hai_pivot h ON h.facility_id::text = x.cms_facility_id;
