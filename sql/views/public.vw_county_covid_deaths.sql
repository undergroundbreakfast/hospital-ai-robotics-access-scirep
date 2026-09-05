CREATE OR REPLACE VIEW public."vw_county_covid_deaths" AS
 SELECT lpad(fips_county_code::text, 5, '0'::text) AS county_fips,
    state,
    county_name,
    urban_rural_code,
    deaths_involving_covid_19
   FROM cdc_covid_deaths;
