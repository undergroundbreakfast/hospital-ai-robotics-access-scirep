CREATE OR REPLACE VIEW public.vw_hospital_county_bridge AS
 WITH crosswalk_src AS (
         SELECT NULLIF(TRIM(BOTH FROM to_jsonb(x.*) ->> 'cms_facility_id'::text), ''::text) AS cms_facility_id_raw,
            NULLIF(TRIM(BOTH FROM to_jsonb(x.*) ->> 'aha_id'::text), ''::text) AS aha_id_raw,
            NULLIF(TRIM(BOTH FROM to_jsonb(x.*) ->> 'aha_county_fips'::text), ''::text) AS aha_county_fips_raw
           FROM aha_cms_crosswalk x
        ), crosswalk_norm AS (
         SELECT NULLIF(regexp_replace(crosswalk_src.cms_facility_id_raw, '\D'::text, ''::text, 'g'::text), ''::text) AS cms_id,
            NULLIF(regexp_replace(crosswalk_src.aha_id_raw, '\D'::text, ''::text, 'g'::text), ''::text) AS aha_id,
            NULLIF(regexp_replace(crosswalk_src.aha_county_fips_raw, '\D'::text, ''::text, 'g'::text), ''::text) AS county_fips
           FROM crosswalk_src
          WHERE crosswalk_src.aha_county_fips_raw IS NOT NULL
        ), cw_by_cms AS (
         SELECT DISTINCT ON (crosswalk_norm.cms_id) crosswalk_norm.cms_id,
            crosswalk_norm.county_fips
           FROM crosswalk_norm
          WHERE crosswalk_norm.cms_id IS NOT NULL AND crosswalk_norm.county_fips IS NOT NULL
          ORDER BY crosswalk_norm.cms_id
        ), cw_by_aha AS (
         SELECT DISTINCT ON (crosswalk_norm.aha_id) crosswalk_norm.aha_id,
            crosswalk_norm.county_fips
           FROM crosswalk_norm
          WHERE crosswalk_norm.aha_id IS NOT NULL AND crosswalk_norm.county_fips IS NOT NULL
          ORDER BY crosswalk_norm.aha_id
        ), aha_base AS (
         SELECT lpad(COALESCE(ahamap.county_fips, cmsmap.county_fips), 5, '0'::text) AS county_fips,
            h.state,
            GREATEST(safe_to_numeric(h.adjpd::text), 0::numeric) AS adjpd,
            safe_to_numeric(h.wfaiart::text) AS wfaiart,
            safe_to_numeric(h.robohos::text) AS robohos,
            safe_to_numeric(h.sep_1::text) AS sep_1,
            safe_to_numeric(h.op_18b::text) AS op_18b,
            safe_to_numeric(h.psi_90_score) AS psi_90_score,
            safe_to_numeric(h.mort_30_pn_score) AS mort_30_pn_score
           FROM vw_hospital_level_aipw h
             LEFT JOIN cw_by_aha ahamap ON ahamap.aha_id = NULLIF(regexp_replace(h.aha_id, '\D'::text, ''::text, 'g'::text), ''::text)
             LEFT JOIN cw_by_cms cmsmap ON cmsmap.cms_id = NULLIF(regexp_replace(h.cms_facility_id, '\D'::text, ''::text, 'g'::text), ''::text)
          WHERE COALESCE(ahamap.county_fips, cmsmap.county_fips) IS NOT NULL
        ), aha_prepped AS (
         SELECT aha_base.county_fips,
            aha_base.state,
            aha_base.adjpd,
            COALESCE(aha_base.wfaiart, 0::numeric) AS wfaiart_val,
            COALESCE(aha_base.robohos, 0::numeric) AS robohos_val,
            aha_base.sep_1,
            aha_base.op_18b,
            aha_base.psi_90_score,
            aha_base.mort_30_pn_score
           FROM aha_base
        ), county_means AS (
         SELECT aha_prepped.county_fips,
            sum(aha_prepped.sep_1 * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.sep_1 IS NOT NULL), 0::numeric) AS sep_1_mean_cty,
            sum(aha_prepped.op_18b * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.op_18b IS NOT NULL), 0::numeric) AS op_18b_mean_cty,
            sum(aha_prepped.psi_90_score * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.psi_90_score IS NOT NULL), 0::numeric) AS psi_90_mean_cty,
            sum(aha_prepped.mort_30_pn_score * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.mort_30_pn_score IS NOT NULL), 0::numeric) AS mort_30_pn_mean_cty
           FROM aha_prepped
          WHERE aha_prepped.adjpd > 0::numeric
          GROUP BY aha_prepped.county_fips
        ), state_means AS (
         SELECT aha_prepped.state,
            sum(aha_prepped.sep_1 * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.sep_1 IS NOT NULL), 0::numeric) AS sep_1_mean_state,
            sum(aha_prepped.op_18b * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.op_18b IS NOT NULL), 0::numeric) AS op_18b_mean_state,
            sum(aha_prepped.psi_90_score * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.psi_90_score IS NOT NULL), 0::numeric) AS psi_90_mean_state,
            sum(aha_prepped.mort_30_pn_score * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.mort_30_pn_score IS NOT NULL), 0::numeric) AS mort_30_pn_mean_state
           FROM aha_prepped
          WHERE aha_prepped.adjpd > 0::numeric
          GROUP BY aha_prepped.state
        ), us_means AS (
         SELECT sum(aha_prepped.sep_1 * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.sep_1 IS NOT NULL), 0::numeric) AS sep_1_mean_us,
            sum(aha_prepped.op_18b * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.op_18b IS NOT NULL), 0::numeric) AS op_18b_mean_us,
            sum(aha_prepped.psi_90_score * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.psi_90_score IS NOT NULL), 0::numeric) AS psi_90_mean_us,
            sum(aha_prepped.mort_30_pn_score * aha_prepped.adjpd) / NULLIF(sum(aha_prepped.adjpd) FILTER (WHERE aha_prepped.mort_30_pn_score IS NOT NULL), 0::numeric) AS mort_30_pn_mean_us
           FROM aha_prepped
          WHERE aha_prepped.adjpd > 0::numeric
        ), aha_imputed AS (
         SELECT a.county_fips,
            a.state,
            a.adjpd,
            a.wfaiart_val,
            a.robohos_val,
            COALESCE(a.sep_1, cm.sep_1_mean_cty, sm.sep_1_mean_state, um.sep_1_mean_us) AS sep_1_imp,
            COALESCE(a.op_18b, cm.op_18b_mean_cty, sm.op_18b_mean_state, um.op_18b_mean_us) AS op_18b_imp,
            COALESCE(a.psi_90_score, cm.psi_90_mean_cty, sm.psi_90_mean_state, um.psi_90_mean_us) AS psi_90_imp,
            COALESCE(a.mort_30_pn_score, cm.mort_30_pn_mean_cty, sm.mort_30_pn_mean_state, um.mort_30_pn_mean_us) AS mort_30_pn_imp
           FROM aha_prepped a
             LEFT JOIN county_means cm ON a.county_fips = cm.county_fips
             LEFT JOIN state_means sm ON a.state = sm.state
             CROSS JOIN us_means um
        ), county_weighted AS (
         SELECT aha_imputed.county_fips,
                CASE
                    WHEN sum(aha_imputed.adjpd) > 0::numeric THEN sum(aha_imputed.wfaiart_val * aha_imputed.adjpd) / sum(aha_imputed.adjpd)
                    ELSE NULL::numeric
                END AS mo14_wfaiart,
                CASE
                    WHEN sum(aha_imputed.adjpd) > 0::numeric THEN sum(aha_imputed.robohos_val * aha_imputed.adjpd) / sum(aha_imputed.adjpd)
                    ELSE NULL::numeric
                END AS mo21_robohos,
                CASE
                    WHEN sum(aha_imputed.adjpd) > 0::numeric THEN sum(aha_imputed.sep_1_imp * aha_imputed.adjpd) / sum(aha_imputed.adjpd)
                    ELSE NULL::numeric
                END AS ef23_sep_1,
                CASE
                    WHEN sum(aha_imputed.adjpd) > 0::numeric THEN sum(aha_imputed.op_18b_imp * aha_imputed.adjpd) / sum(aha_imputed.adjpd)
                    ELSE NULL::numeric
                END AS ef6_op_18b,
                CASE
                    WHEN sum(aha_imputed.adjpd) > 0::numeric THEN sum(aha_imputed.psi_90_imp * aha_imputed.adjpd) / sum(aha_imputed.adjpd)
                    ELSE NULL::numeric
                END AS fa21_psi_90,
                CASE
                    WHEN sum(aha_imputed.adjpd) > 0::numeric THEN sum(aha_imputed.mort_30_pn_imp * aha_imputed.adjpd) / sum(aha_imputed.adjpd)
                    ELSE NULL::numeric
                END AS fa27_mort_30_pn
           FROM aha_imputed
          WHERE aha_imputed.adjpd > 0::numeric
          GROUP BY aha_imputed.county_fips
        ), chr_county AS (
         SELECT lpad(COALESCE(c1._5_digit_fips, c2._5_digit_fips)::text, 5, '0'::text) AS county_fips,
            safe_to_numeric(c1.premature_death_raw_value::text) AS dv21_ypll_chr
           FROM chr_analytic_chunk_1 c1
             FULL JOIN chr_analytic_chunk_2 c2 ON c1._5_digit_fips::text = c2._5_digit_fips::text
        ), cdc_ct5 AS (
         SELECT lpad("vw_2023_age_adj_YPLL".county_fips, 5, '0'::text) AS county_fips,
            safe_to_numeric("vw_2023_age_adj_YPLL".ct5_ypll_u75_age_adj_per_100k_mid::text) AS ct5_ypll_u75_age_adj_per_100k_mid
           FROM "vw_2023_age_adj_YPLL"
          WHERE "vw_2023_age_adj_YPLL".county_fips IS NOT NULL
        ), county_universe AS (
         SELECT COALESCE(chr.county_fips, ct5.county_fips) AS county_fips,
            chr.dv21_ypll_chr,
            ct5.ct5_ypll_u75_age_adj_per_100k_mid
           FROM chr_county chr
             FULL JOIN cdc_ct5 ct5 USING (county_fips)
        )
 SELECT u.county_fips,
    cw.mo14_wfaiart,
    cw.mo21_robohos,
    u.dv21_ypll_chr,
    u.ct5_ypll_u75_age_adj_per_100k_mid,
    cw.ef23_sep_1,
    cw.ef6_op_18b,
    cw.fa21_psi_90,
    cw.fa27_mort_30_pn
   FROM county_universe u
     LEFT JOIN county_weighted cw ON u.county_fips = cw.county_fips
  WHERE u.county_fips !~~ '%000'::text
  ORDER BY u.county_fips;;
