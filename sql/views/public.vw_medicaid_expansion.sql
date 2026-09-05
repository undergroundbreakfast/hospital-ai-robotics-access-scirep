CREATE OR REPLACE VIEW public."vw_medicaid_expansion" AS
 SELECT lpad(hhd.county_fips_code::text, 5, '0'::text) AS county_fips_code,
    hhd.state_name,
    hhd.irr_county_value,
    imd.val AS mortality_value,
    kme.exp_active_2019
   FROM hrsa_health_equity_data hhd
     LEFT JOIN kff_medicaid_expansion_012625 kme ON hhd.state_name::text = kme.state_name::text
     LEFT JOIN ihme_mortality_data_county_level_all_types_2019 imd ON lpad(hhd.county_fips_code::text, 5, '0'::text) = lpad(imd.fips::text, 5, '0'::text)
  WHERE hhd.irr_county_value IS NOT NULL;;
