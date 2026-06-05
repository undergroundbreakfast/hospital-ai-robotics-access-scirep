CREATE TABLE IF NOT EXISTS public.cms_2022_hospital_quality_long (
    snapshot text NOT NULL,
    source_archive text NOT NULL,
    source_file text NOT NULL,
    facility_id text NOT NULL,
    facility_name text,
    address text,
    city text,
    state text,
    zip_code text,
    county_name text,
    phone_number text,
    hospital_type text,
    hospital_ownership text,
    emergency_services text,
    condition text,
    measure_id text NOT NULL,
    measure_name text,
    score_raw text,
    score_numeric numeric,
    sample text,
    denominator text,
    compared_to_national text,
    lower_estimate text,
    higher_estimate text,
    footnote text,
    row_start_date text,
    row_end_date text,
    measure_start_quarter text,
    measure_start_date text,
    measure_end_quarter text,
    measure_end_date text,
    PRIMARY KEY (snapshot, measure_id, facility_id)
);

CREATE TEMP TABLE cms_2022_hospital_quality_load
(LIKE public.cms_2022_hospital_quality_long INCLUDING DEFAULTS)
ON COMMIT PRESERVE ROWS;

\if :{?cms_2022_csv}
\else
  \set cms_2022_csv 'data/public/cms_2022_hospital_quality/filtered/cms_2022_hospital_quality_long.csv'
\endif

\copy cms_2022_hospital_quality_load (snapshot, source_archive, source_file, facility_id, facility_name, address, city, state, zip_code, county_name, phone_number, hospital_type, hospital_ownership, emergency_services, condition, measure_id, measure_name, score_raw, score_numeric, sample, denominator, compared_to_national, lower_estimate, higher_estimate, footnote, row_start_date, row_end_date, measure_start_quarter, measure_start_date, measure_end_quarter, measure_end_date) FROM :'cms_2022_csv' WITH (FORMAT csv, HEADER true, NULL '');

INSERT INTO public.cms_2022_hospital_quality_long (
    snapshot,
    source_archive,
    source_file,
    facility_id,
    facility_name,
    address,
    city,
    state,
    zip_code,
    county_name,
    phone_number,
    hospital_type,
    hospital_ownership,
    emergency_services,
    condition,
    measure_id,
    measure_name,
    score_raw,
    score_numeric,
    sample,
    denominator,
    compared_to_national,
    lower_estimate,
    higher_estimate,
    footnote,
    row_start_date,
    row_end_date,
    measure_start_quarter,
    measure_start_date,
    measure_end_quarter,
    measure_end_date
)
SELECT
    snapshot,
    source_archive,
    source_file,
    facility_id,
    facility_name,
    address,
    city,
    state,
    zip_code,
    county_name,
    phone_number,
    hospital_type,
    hospital_ownership,
    emergency_services,
    condition,
    measure_id,
    measure_name,
    score_raw,
    score_numeric,
    sample,
    denominator,
    compared_to_national,
    lower_estimate,
    higher_estimate,
    footnote,
    row_start_date,
    row_end_date,
    measure_start_quarter,
    measure_start_date,
    measure_end_quarter,
    measure_end_date
FROM cms_2022_hospital_quality_load
ON CONFLICT (snapshot, measure_id, facility_id) DO UPDATE SET
    source_archive = EXCLUDED.source_archive,
    source_file = EXCLUDED.source_file,
    facility_name = EXCLUDED.facility_name,
    address = EXCLUDED.address,
    city = EXCLUDED.city,
    state = EXCLUDED.state,
    zip_code = EXCLUDED.zip_code,
    county_name = EXCLUDED.county_name,
    phone_number = EXCLUDED.phone_number,
    hospital_type = EXCLUDED.hospital_type,
    hospital_ownership = EXCLUDED.hospital_ownership,
    emergency_services = EXCLUDED.emergency_services,
    condition = EXCLUDED.condition,
    measure_name = EXCLUDED.measure_name,
    score_raw = EXCLUDED.score_raw,
    score_numeric = EXCLUDED.score_numeric,
    sample = EXCLUDED.sample,
    denominator = EXCLUDED.denominator,
    compared_to_national = EXCLUDED.compared_to_national,
    lower_estimate = EXCLUDED.lower_estimate,
    higher_estimate = EXCLUDED.higher_estimate,
    footnote = EXCLUDED.footnote,
    row_start_date = EXCLUDED.row_start_date,
    row_end_date = EXCLUDED.row_end_date,
    measure_start_quarter = EXCLUDED.measure_start_quarter,
    measure_start_date = EXCLUDED.measure_start_date,
    measure_end_quarter = EXCLUDED.measure_end_quarter,
    measure_end_date = EXCLUDED.measure_end_date;

CREATE INDEX IF NOT EXISTS idx_cms_2022_quality_facility
    ON public.cms_2022_hospital_quality_long (facility_id);

CREATE INDEX IF NOT EXISTS idx_cms_2022_quality_measure_snapshot
    ON public.cms_2022_hospital_quality_long (measure_id, snapshot);

CREATE OR REPLACE VIEW public.vw_cms_2022_hospital_quality_scored AS
SELECT *
FROM public.cms_2022_hospital_quality_long
WHERE score_numeric IS NOT NULL;
