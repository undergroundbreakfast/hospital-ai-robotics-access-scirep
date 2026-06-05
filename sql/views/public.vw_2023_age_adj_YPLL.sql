CREATE OR REPLACE VIEW public.vw_2023_age_adj_YPLL AS
 WITH pop_wide AS (
         SELECT lpad(census_est2024_population.state::text, 2, '0'::text) || lpad(census_est2024_population.county::text, 3, '0'::text) AS county_fips,
            census_est2024_population.under5_tot::numeric AS pop_0_4,
            census_est2024_population.age513_tot::numeric AS pop_5_14,
            census_est2024_population.age1417_tot::numeric + census_est2024_population.age1824_tot::numeric AS pop_15_24,
            census_est2024_population.age2529_tot::numeric + census_est2024_population.age3034_tot::numeric AS pop_25_34,
            census_est2024_population.age3539_tot::numeric + census_est2024_population.age4044_tot::numeric AS pop_35_44,
            census_est2024_population.age4549_tot::numeric + census_est2024_population.age5054_tot::numeric AS pop_45_54,
            census_est2024_population.age5559_tot::numeric + census_est2024_population.age6064_tot::numeric AS pop_55_64,
            census_est2024_population.age6569_tot::numeric + census_est2024_population.age7074_tot::numeric AS pop_65_74,
            census_est2024_population.popestimate::numeric AS pop_total
           FROM census_est2024_population
          WHERE census_est2024_population.year::text = '6'::text
        ), pop_long AS (
         SELECT pop_wide.county_fips,
            '0-4'::text AS band,
            pop_wide.pop_0_4 AS pop
           FROM pop_wide
        UNION ALL
         SELECT pop_wide.county_fips,
            '5-14'::text,
            pop_wide.pop_5_14
           FROM pop_wide
        UNION ALL
         SELECT pop_wide.county_fips,
            '15-24'::text,
            pop_wide.pop_15_24
           FROM pop_wide
        UNION ALL
         SELECT pop_wide.county_fips,
            '25-34'::text,
            pop_wide.pop_25_34
           FROM pop_wide
        UNION ALL
         SELECT pop_wide.county_fips,
            '35-44'::text,
            pop_wide.pop_35_44
           FROM pop_wide
        UNION ALL
         SELECT pop_wide.county_fips,
            '45-54'::text,
            pop_wide.pop_45_54
           FROM pop_wide
        UNION ALL
         SELECT pop_wide.county_fips,
            '55-64'::text,
            pop_wide.pop_55_64
           FROM pop_wide
        UNION ALL
         SELECT pop_wide.county_fips,
            '65-74'::text,
            pop_wide.pop_65_74
           FROM pop_wide
        ), pop_u75 AS (
         SELECT pop_wide.county_fips,
            COALESCE(pop_wide.pop_0_4, 0::numeric) + COALESCE(pop_wide.pop_5_14, 0::numeric) + COALESCE(pop_wide.pop_15_24, 0::numeric) + COALESCE(pop_wide.pop_25_34, 0::numeric) + COALESCE(pop_wide.pop_35_44, 0::numeric) + COALESCE(pop_wide.pop_45_54, 0::numeric) + COALESCE(pop_wide.pop_55_64, 0::numeric) + COALESCE(pop_wide.pop_65_74, 0::numeric) AS pop_under_75,
            pop_wide.pop_total
           FROM pop_wide
        ), std_weights AS (
         SELECT t.band,
            t.w
           FROM ( VALUES ('0-4'::text,0.07357414491987505), ('5-14'::text,0.15491230408271922), ('15-24'::text,0.14755014650177273), ('25-34'::text,0.14428227446119865), ('35-44'::text,0.17305758765658530), ('45-54'::text,0.14349453031166326), ('55-64'::text,0.09285257724651255), ('65-74'::text,0.07027643481967326)) t(band, w)
        ), cdc_deaths_raw AS (
         SELECT lpad(TRIM(BOTH FROM cdc_2023_age_banded_deaths_county.county_fips::text), 5, '0'::text) AS county_fips,
            cdc_2023_age_banded_deaths_county.residence_county,
                CASE
                    WHEN cdc_2023_age_banded_deaths_county.ten_year_age_group::text = ANY (ARRAY['1'::character varying, '<1'::character varying, 'Under 1'::character varying, 'Under 1 year'::character varying, 'Under 1 years'::character varying]::text[]) THEN '<1'::character varying
                    ELSE cdc_2023_age_banded_deaths_county.ten_year_age_group
                END AS age_band,
            NULLIF(TRIM(BOTH FROM cdc_2023_age_banded_deaths_county.deaths::text), ''::text) AS deaths_txt
           FROM cdc_2023_age_banded_deaths_county
        ), cdc_deaths_typed AS (
         SELECT cdc_deaths_raw.county_fips,
            cdc_deaths_raw.residence_county,
            cdc_deaths_raw.age_band,
                CASE
                    WHEN cdc_deaths_raw.deaths_txt ~ '^[0-9]+$'::text THEN cdc_deaths_raw.deaths_txt::integer
                    ELSE NULL::integer
                END AS deaths_n,
                CASE
                    WHEN cdc_deaths_raw.deaths_txt IS NULL THEN 1
                    ELSE 0
                END AS is_suppressed
           FROM cdc_deaths_raw
        ), midpoints AS (
         SELECT t.age_band,
            t.midpoint,
            t.pop_band
           FROM ( VALUES ('<1'::text,0.5,'0-4'::text), ('1-4'::text,2.5,'0-4'::text), ('5-14'::text,9.5,'5-14'::text), ('15-24'::text,19.5,'15-24'::text), ('25-34'::text,29.5,'25-34'::text), ('35-44'::text,39.5,'35-44'::text), ('45-54'::text,49.5,'45-54'::text), ('55-64'::text,59.5,'55-64'::text), ('65-74'::text,69.5,'65-74'::text)) t(age_band, midpoint, pop_band)
        ), ypll_by_county_band AS (
         SELECT d.county_fips,
            max(d.residence_county::text) AS residence_county,
            m.pop_band AS band,
            sum(d.is_suppressed) AS suppressed_cells_included,
            sum(COALESCE(d.deaths_n, 1)::numeric * GREATEST(0::numeric, 75::numeric - m.midpoint)) AS ypll_low,
            sum(COALESCE(d.deaths_n, 5)::numeric * GREATEST(0::numeric, 75::numeric - m.midpoint)) AS ypll_mid,
            sum(COALESCE(d.deaths_n, 9)::numeric * GREATEST(0::numeric, 75::numeric - m.midpoint)) AS ypll_high
           FROM cdc_deaths_typed d
             JOIN midpoints m ON d.age_band::text = m.age_band
          WHERE d.age_band::text = ANY (ARRAY['<1'::character varying, '1-4'::character varying, '5-14'::character varying, '15-24'::character varying, '25-34'::character varying, '35-44'::character varying, '45-54'::character varying, '55-64'::character varying, '65-74'::character varying]::text[])
          GROUP BY d.county_fips, m.pop_band
        ), ypll_rates AS (
         SELECT y.county_fips,
            y.residence_county,
            y.band,
            y.suppressed_cells_included,
            p.pop,
            y.ypll_low / NULLIF(p.pop, 0::numeric) * 100000.0 AS ypll_rate_low,
            y.ypll_mid / NULLIF(p.pop, 0::numeric) * 100000.0 AS ypll_rate_mid,
            y.ypll_high / NULLIF(p.pop, 0::numeric) * 100000.0 AS ypll_rate_high
           FROM ypll_by_county_band y
             JOIN pop_long p ON p.county_fips = y.county_fips AND p.band = y.band
        ), ct5_age_adjusted AS (
         SELECT r.county_fips,
            max(r.residence_county) AS residence_county,
            sum(r.suppressed_cells_included) AS suppressed_cells_total,
            sum(w.w * r.ypll_rate_low) AS ct5_ypll_u75_age_adj_per_100k_low,
            sum(w.w * r.ypll_rate_mid) AS ct5_ypll_u75_age_adj_per_100k_mid,
            sum(w.w * r.ypll_rate_high) AS ct5_ypll_u75_age_adj_per_100k_high
           FROM ypll_rates r
             JOIN std_weights w ON w.band = r.band
          GROUP BY r.county_fips
        ), ct5_crude AS (
         SELECT b.county_fips,
            sum(b.ypll_low) AS total_ypll_low,
            sum(b.ypll_mid) AS total_ypll_mid,
            sum(b.ypll_high) AS total_ypll_high
           FROM ypll_by_county_band b
          GROUP BY b.county_fips
        )
 SELECT a.county_fips,
    a.residence_county,
    a.suppressed_cells_total,
    pu.pop_under_75,
    pu.pop_total,
    a.ct5_ypll_u75_age_adj_per_100k_low,
    a.ct5_ypll_u75_age_adj_per_100k_mid,
    a.ct5_ypll_u75_age_adj_per_100k_high,
    c.total_ypll_low / NULLIF(pu.pop_under_75, 0::numeric) * 100000.0 AS ct5_ypll_u75_crude_per_100k_low,
    c.total_ypll_mid / NULLIF(pu.pop_under_75, 0::numeric) * 100000.0 AS ct5_ypll_u75_crude_per_100k_mid,
    c.total_ypll_high / NULLIF(pu.pop_under_75, 0::numeric) * 100000.0 AS ct5_ypll_u75_crude_per_100k_high
   FROM ct5_age_adjusted a
     LEFT JOIN ct5_crude c ON c.county_fips = a.county_fips
     LEFT JOIN pop_u75 pu ON pu.county_fips = a.county_fips
  ORDER BY a.county_fips;;
