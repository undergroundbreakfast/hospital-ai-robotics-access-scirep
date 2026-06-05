CREATE OR REPLACE VIEW public.vw_capex_per_sqfoot AS
 SELECT a.id,
    a.fcounty,
    a.mname,
    NULLIF(a.gfeet::text, ''::text)::numeric AS square_footage,
    NULLIF(a.ceamt::text, ''::text)::numeric AS capital_expenditure,
    NULLIF(a.ceamt::text, ''::text)::numeric / NULLIF(a.gfeet::text, ''::text)::numeric AS capex_per_sqft,
    u.state_name
   FROM aha_survey_data a
     LEFT JOIN uscounties u ON lpad(a.fcounty::text, 5, '0'::text) = u.county_fips::text
  WHERE a.gfeet IS NOT NULL AND a.ceamt IS NOT NULL
  ORDER BY (NULLIF(a.ceamt::text, ''::text)::numeric) DESC;;
