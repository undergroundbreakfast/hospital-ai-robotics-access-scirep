CREATE OR REPLACE VIEW public."vw_2023_hospital_deaths" AS
 WITH params AS (
         SELECT 5::numeric AS suppressed_impute_value,
            100000.0 AS rate_multiplier
        ), pop_wide AS (
         SELECT lpad(c.state::text, 2, '0'::text) || lpad(c.county::text, 3, '0'::text) AS county_fips,
            c.under5_tot::numeric AS pop_0_4,
            c.age513_tot::numeric AS pop_5_14,
            c.age1417_tot::numeric + c.age1824_tot::numeric AS pop_15_24,
            c.age2529_tot::numeric + c.age3034_tot::numeric AS pop_25_34,
            c.age3539_tot::numeric + c.age4044_tot::numeric AS pop_35_44,
            c.age4549_tot::numeric + c.age5054_tot::numeric AS pop_45_54,
            c.age5559_tot::numeric + c.age6064_tot::numeric AS pop_55_64,
            c.age6569_tot::numeric + c.age7074_tot::numeric AS pop_65_74,
            c.popestimate::numeric AS pop_total
           FROM census_est2024_population c
          WHERE c.year::text = '6'::text
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
        ), std_weights AS (
         SELECT t.band,
            t.w
           FROM ( VALUES ('0-4'::text,0.07357414491987505), ('5-14'::text,0.15491230408271922), ('15-24'::text,0.14755014650177273), ('25-34'::text,0.14428227446119865), ('35-44'::text,0.17305758765658530), ('45-54'::text,0.14349453031166326), ('55-64'::text,0.09285257724651255), ('65-74'::text,0.07027643481967326)) t(band, w)
        ), bands AS (
         SELECT v.band
           FROM ( VALUES ('0-4'::text), ('5-14'::text), ('15-24'::text), ('25-34'::text), ('35-44'::text), ('45-54'::text), ('55-64'::text), ('65-74'::text)) v(band)
        ), cdc_deaths_raw AS (
         SELECT lpad(TRIM(BOTH FROM d.county_fips::text), 5, '0'::text) AS county_fips,
            d.residence_county,
            TRIM(BOTH FROM d.ten_year_age_group::text) AS age_group_raw,
            NULLIF(TRIM(BOTH FROM d.deaths::text), ''::text) AS deaths_txt
           FROM cdc_2023_age_banded_hospital_deaths d
        ), cdc_deaths_typed AS (
         SELECT r.county_fips,
            r.residence_county,
                CASE
                    WHEN r.age_group_raw = ANY (ARRAY['1'::text, '<1'::text, 'Under 1'::text, 'Under 1 year'::text, 'Under 1 years'::text]) THEN '<1'::text
                    ELSE r.age_group_raw
                END AS age_band,
                CASE
                    WHEN r.deaths_txt ~ '^[0-9]+$'::text THEN r.deaths_txt::integer
                    ELSE NULL::integer
                END AS deaths_n,
                CASE
                    WHEN r.deaths_txt IS NULL THEN 1
                    ELSE 0
                END AS is_suppressed
           FROM cdc_deaths_raw r
        ), county_has_0_4 AS (
         SELECT cdc_deaths_typed.county_fips,
            max(
                CASE
                    WHEN cdc_deaths_typed.age_band = '0-4'::text THEN 1
                    ELSE 0
                END) AS has_0_4
           FROM cdc_deaths_typed
          GROUP BY cdc_deaths_typed.county_fips
        ), deaths_mapped AS (
         SELECT d.county_fips,
            d.residence_county,
                CASE
                    WHEN d.age_band = ANY (ARRAY['0-4'::text, '<1'::text, '1-4'::text]) THEN '0-4'::text
                    WHEN d.age_band = '5-14'::text THEN '5-14'::text
                    WHEN d.age_band = '15-24'::text THEN '15-24'::text
                    WHEN d.age_band = '25-34'::text THEN '25-34'::text
                    WHEN d.age_band = '35-44'::text THEN '35-44'::text
                    WHEN d.age_band = '45-54'::text THEN '45-54'::text
                    WHEN d.age_band = '55-64'::text THEN '55-64'::text
                    WHEN d.age_band = '65-74'::text THEN '65-74'::text
                    ELSE NULL::text
                END AS band,
            d.age_band,
            d.is_suppressed,
                CASE
                    WHEN d.is_suppressed = 1 THEN ( SELECT params.suppressed_impute_value
                       FROM params)
                    ELSE d.deaths_n::numeric
                END AS deaths_imputed
           FROM cdc_deaths_typed d
             JOIN county_has_0_4 h ON h.county_fips = d.county_fips
          WHERE (d.age_band = ANY (ARRAY['0-4'::text, '<1'::text, '1-4'::text, '5-14'::text, '15-24'::text, '25-34'::text, '35-44'::text, '45-54'::text, '55-64'::text, '65-74'::text])) AND NOT (h.has_0_4 = 1 AND (d.age_band = ANY (ARRAY['<1'::text, '1-4'::text])))
        ), deaths_by_county_band AS (
         SELECT deaths_mapped.county_fips,
            deaths_mapped.band,
            sum(deaths_mapped.deaths_imputed) AS deaths
           FROM deaths_mapped
          WHERE deaths_mapped.band IS NOT NULL
          GROUP BY deaths_mapped.county_fips, deaths_mapped.band
        ), counties AS (
         SELECT DISTINCT deaths_by_county_band.county_fips
           FROM deaths_by_county_band
        ), county_band_grid AS (
         SELECT c.county_fips,
            b.band
           FROM counties c
             CROSS JOIN bands b
        ), deaths_filled AS (
         SELECT g.county_fips,
            g.band,
            COALESCE(d.deaths, 0::numeric) AS deaths
           FROM county_band_grid g
             LEFT JOIN deaths_by_county_band d ON d.county_fips = g.county_fips AND d.band = g.band
        ), pivot_counts AS (
         SELECT deaths_filled.county_fips,
            sum(deaths_filled.deaths) FILTER (WHERE deaths_filled.band = '0-4'::text) AS total_0_4,
            sum(deaths_filled.deaths) FILTER (WHERE deaths_filled.band = '5-14'::text) AS total_5_14,
            sum(deaths_filled.deaths) FILTER (WHERE deaths_filled.band = '15-24'::text) AS total_15_24,
            sum(deaths_filled.deaths) FILTER (WHERE deaths_filled.band = '25-34'::text) AS total_25_34,
            sum(deaths_filled.deaths) FILTER (WHERE deaths_filled.band = '35-44'::text) AS total_35_44,
            sum(deaths_filled.deaths) FILTER (WHERE deaths_filled.band = '45-54'::text) AS total_45_54,
            sum(deaths_filled.deaths) FILTER (WHERE deaths_filled.band = '55-64'::text) AS total_55_64,
            sum(deaths_filled.deaths) FILTER (WHERE deaths_filled.band = '65-74'::text) AS total_65_74,
            sum(deaths_filled.deaths) AS total_raw_hospital_deaths
           FROM deaths_filled
          GROUP BY deaths_filled.county_fips
        ), death_rates AS (
         SELECT f.county_fips,
            f.band,
            f.deaths,
            p.pop,
            f.deaths / NULLIF(p.pop, 0::numeric) * (( SELECT params.rate_multiplier
                   FROM params)) AS rate_per_100k
           FROM deaths_filled f
             LEFT JOIN pop_long p ON p.county_fips = f.county_fips AND p.band = f.band
        ), age_adjusted AS (
         SELECT r.county_fips,
            sum(w.w * r.rate_per_100k) AS total_age_adjusted_deaths
           FROM death_rates r
             JOIN std_weights w ON w.band = r.band
          GROUP BY r.county_fips
        )
 SELECT pc.county_fips,
    pc.total_raw_hospital_deaths,
    aa.total_age_adjusted_deaths,
    pc.total_0_4,
    pc.total_5_14,
    pc.total_15_24,
    pc.total_25_34,
    pc.total_35_44,
    pc.total_45_54,
    pc.total_55_64,
    pc.total_65_74
   FROM pivot_counts pc
     LEFT JOIN age_adjusted aa ON aa.county_fips = pc.county_fips
  ORDER BY pc.county_fips;;
