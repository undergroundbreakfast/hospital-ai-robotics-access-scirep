CREATE OR REPLACE VIEW public.vw_irr_cdc_wonder_mortality_2023 AS
 SELECT m.county,
    m.county_code,
    m.deaths,
    m.population,
    m.crude_rate::numeric AS crude_rate_numeric,
    h.state_name,
    h.irr_county_value::numeric AS irr_numeric
   FROM cdc_wonder_5_year_mortality_2023 m
     JOIN hrsa_health_equity_data h ON lpad(m.county_code::text, 5, '0'::text) = lpad(h.county_fips_code::text, 5, '0'::text)
  WHERE m.crude_rate::text <> 'Unreliable'::text AND m.crude_rate::text <> 'Not Available'::text
  ORDER BY (m.crude_rate::numeric) DESC;;
