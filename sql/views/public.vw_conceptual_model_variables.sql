CREATE OR REPLACE VIEW public."vw_conceptual_model_variables" AS
 WITH hospital_data AS (
         SELECT lpad(h.county_fips, 5, '0'::text) AS county_fips,
            safe_to_numeric(h.ai_adoption_score::text) AS ai_adoption_score,
            safe_to_numeric(r.robotics_adoption_score::text) AS robotics_adoption_score,
            safe_to_numeric(NULLIF(regexp_replace(TRIM(BOTH FROM ps.patient_services_margin), '%'::text, ''::text, 'g'::text), ''::text)) AS patient_services_margin,
            safe_to_numeric(b_1.total_beds::text) AS total_beds
           FROM vw_hospital_ai_score h
             LEFT JOIN vw_robotics_score_by_hospital r ON lpad(h.county_fips, 5, '0'::text) = lpad(r.county_fips, 5, '0'::text)
             LEFT JOIN matched_hospitals_updated ps ON lpad(h.county_fips, 5, '0'::text) = lpad(ps.aha_county_fips::text, 5, '0'::text)
             LEFT JOIN vw_tech_enabled_beds_by_county b_1 ON lpad(h.county_fips, 5, '0'::text) = lpad(b_1.county_fips, 5, '0'::text)
        ), weighted_hospital_data AS (
         SELECT hospital_data.county_fips,
            COALESCE(
                CASE
                    WHEN sum(hospital_data.total_beds) > 0::numeric THEN sum(hospital_data.ai_adoption_score * hospital_data.total_beds) / sum(hospital_data.total_beds)
                    ELSE NULL::numeric
                END, 0::numeric) AS weighted_ai_adoption_score,
            COALESCE(
                CASE
                    WHEN sum(hospital_data.total_beds) > 0::numeric THEN sum(hospital_data.robotics_adoption_score * hospital_data.total_beds) / sum(hospital_data.total_beds)
                    ELSE NULL::numeric
                END, 0::numeric) AS weighted_robotics_adoption_score
           FROM hospital_data
          GROUP BY hospital_data.county_fips
        ), raw_data AS (
         SELECT COALESCE(c1._5_digit_fips, c2._5_digit_fips) AS county_fips,
            safe_to_numeric(c1.premature_death_raw_value::text) AS premature_death_raw_value,
            safe_to_numeric(c1.poor_or_fair_health_raw_value::text) AS poor_or_fair_health_raw_value,
            safe_to_numeric(c1.poor_physical_health_days_raw_value::text) AS poor_physical_health_days_raw_value,
            safe_to_numeric(c1.poor_mental_health_days_raw_value::text) AS poor_mental_health_days_raw_value,
            safe_to_numeric(c1.low_birthweight_raw_value::text) AS low_birthweight_raw_value,
            safe_to_numeric(c1.adult_smoking_raw_value::text) AS adult_smoking_raw_value,
            safe_to_numeric(c1.adult_obesity_raw_value::text) AS adult_obesity_raw_value,
            safe_to_numeric(c1.food_environment_index_raw_value::text) AS food_environment_index_raw_value,
            safe_to_numeric(c1.physical_inactivity_raw_value::text) AS physical_inactivity_raw_value,
            safe_to_numeric(c1.access_to_exercise_opportunities_raw_value::text) AS access_to_exercise_opportunities_raw_value,
            safe_to_numeric(c1.excessive_drinking_raw_value::text) AS excessive_drinking_raw_value,
            safe_to_numeric(c1.alcohol_impaired_driving_deaths_raw_value::text) AS alcohol_impaired_driving_deaths_raw_value,
            safe_to_numeric(c1.sexually_transmitted_infections_raw_value::text) AS sexually_transmitted_infections_raw_value,
            safe_to_numeric(c1.teen_births_raw_value::text) AS teen_births_raw_value,
            safe_to_numeric(c1.uninsured_raw_value::text) AS uninsured_raw_value,
            safe_to_numeric(c1.ratio_of_population_to_primary_care_physicians::text) AS ratio_of_population_to_primary_care_physicians,
            safe_to_numeric(c1.ratio_of_population_to_dentists::text) AS ratio_of_population_to_dentists,
            safe_to_numeric(c1.ratio_of_population_to_mental_health_providers::text) AS ratio_of_population_to_mental_health_providers,
            safe_to_numeric(c1.preventable_hospital_stays_raw_value::text) AS preventable_hospital_stays_raw_value,
            safe_to_numeric(c1.mammography_screening_raw_value::text) AS mammography_screening_raw_value,
            safe_to_numeric(c1.flu_vaccinations_raw_value::text) AS flu_vaccinations_raw_value,
            safe_to_numeric(c1.high_school_completion_raw_value::text) AS high_school_completion_raw_value,
            safe_to_numeric(c1.some_college_raw_value::text) AS some_college_raw_value,
            safe_to_numeric(c1.unemployment_raw_value::text) AS unemployment_raw_value,
            safe_to_numeric(c1.children_in_poverty_raw_value::text) AS children_in_poverty_raw_value,
            safe_to_numeric(c2.air_pollution_particulate_matter_raw_value::text) AS air_pollution_particulate_matter_raw_value,
            safe_to_numeric(c2.drinking_water_violations_raw_value::text) AS drinking_water_violations_raw_value,
            safe_to_numeric(c2.severe_housing_problems_raw_value::text) AS severe_housing_problems_raw_value,
            safe_to_numeric(c2.driving_alone_to_work_raw_value::text) AS driving_alone_to_work_raw_value,
            safe_to_numeric(c2.long_commute_driving_alone_raw_value::text) AS long_commute_driving_alone_raw_value,
            safe_to_numeric(c2.income_inequality_raw_value::text) AS income_inequality_raw_value,
            safe_to_numeric(c2.children_in_single_parent_households_raw_value::text) AS children_in_single_parent_households_raw_value,
            safe_to_numeric(c2.social_associations_raw_value::text) AS social_associations_raw_value,
            safe_to_numeric(c2.injury_deaths_raw_value::text) AS injury_deaths_raw_value
           FROM chr_analytic_chunk_1 c1
             FULL JOIN chr_analytic_chunk_2 c2 ON c1._5_digit_fips::text = c2._5_digit_fips::text
        )
 SELECT rd.county_fips,
    rd.premature_death_raw_value,
    rd.poor_or_fair_health_raw_value,
    rd.poor_physical_health_days_raw_value,
    rd.poor_mental_health_days_raw_value,
    rd.low_birthweight_raw_value,
    rd.adult_smoking_raw_value,
    rd.adult_obesity_raw_value,
    rd.food_environment_index_raw_value,
    rd.physical_inactivity_raw_value,
    rd.access_to_exercise_opportunities_raw_value,
    rd.excessive_drinking_raw_value,
    rd.alcohol_impaired_driving_deaths_raw_value,
    rd.sexually_transmitted_infections_raw_value,
    rd.teen_births_raw_value,
    rd.uninsured_raw_value,
    rd.ratio_of_population_to_primary_care_physicians,
    rd.ratio_of_population_to_dentists,
    rd.ratio_of_population_to_mental_health_providers,
    rd.preventable_hospital_stays_raw_value,
    rd.mammography_screening_raw_value,
    rd.flu_vaccinations_raw_value,
    rd.high_school_completion_raw_value,
    rd.some_college_raw_value,
    rd.unemployment_raw_value,
    rd.children_in_poverty_raw_value,
    rd.air_pollution_particulate_matter_raw_value,
    rd.drinking_water_violations_raw_value,
    rd.severe_housing_problems_raw_value,
    rd.driving_alone_to_work_raw_value,
    rd.long_commute_driving_alone_raw_value,
    rd.income_inequality_raw_value,
    rd.children_in_single_parent_households_raw_value,
    rd.social_associations_raw_value,
    rd.injury_deaths_raw_value,
    whd.weighted_ai_adoption_score,
    whd.weighted_robotics_adoption_score
   FROM raw_data rd
     LEFT JOIN weighted_hospital_data whd ON rd.county_fips::text = whd.county_fips
  WHERE rd.county_fips::text !~~ '%000'::text
  ORDER BY rd.county_fips;
