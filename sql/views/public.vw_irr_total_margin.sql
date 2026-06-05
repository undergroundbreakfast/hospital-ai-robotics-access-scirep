CREATE OR REPLACE VIEW public.vw_irr_total_margin AS
 SELECT mh.aha_id,
    mh.total_margin AS dv,
    hhe.irr_county_value AS iv,
    mh.aha_county_fips,
    hhe.county_fips_code
   FROM matched_hospitals mh
     JOIN hrsa_health_equity_data hhe ON lpad(mh.aha_county_fips::text, 5, '0'::text) = lpad(hhe.county_fips_code::text, 5, '0'::text);;
