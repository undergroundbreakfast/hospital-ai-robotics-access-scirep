CREATE OR REPLACE VIEW public."vw_hospital_ownership_percentages" AS
 SELECT s.fcounty AS county_fips,
    s.cntrl,
    o.category,
    o.description,
    s.total_adjpd_own,
    s.pct_adjpd_own_in_county
   FROM ( SELECT aha_survey_data.fcounty,
            aha_survey_data.cntrl,
            sum(aha_survey_data.adjpd::numeric) AS total_adjpd_own,
            sum(aha_survey_data.adjpd::numeric) / NULLIF(sum(sum(aha_survey_data.adjpd::numeric)) OVER (PARTITION BY aha_survey_data.fcounty), 0::numeric) AS pct_adjpd_own_in_county
           FROM aha_survey_data
          GROUP BY aha_survey_data.fcounty, aha_survey_data.cntrl) s
     JOIN aha_appendix_a_ownership_codes o ON s.cntrl::text = o.code::text
  ORDER BY s.fcounty, s.cntrl;
