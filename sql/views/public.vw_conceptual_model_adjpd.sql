CREATE OR REPLACE VIEW public."vw_conceptual_model_adjpd" AS
 WITH hospital_data AS (
         SELECT lpad(h.county_fips, 5, '0'::text) AS county_fips,
            safe_to_numeric(h.ai_adoption_score::text) AS ai_adoption_score,
            safe_to_numeric(r.robotics_adoption_score::text) AS robotics_adoption_score,
            safe_to_numeric(NULLIF(regexp_replace(TRIM(BOTH FROM ps.patient_services_margin), '%'::text, ''::text, 'g'::text), ''::text)) AS patient_services_margin,
            safe_to_numeric(b_1.total_adjpd::text) AS total_adjpd
           FROM vw_hospital_ai_score_adjpd h
             LEFT JOIN vw_robotics_score_by_hospital_adjpd r ON lpad(h.county_fips, 5, '0'::text) = lpad(r.county_fips, 5, '0'::text)
             LEFT JOIN matched_hospitals_updated ps ON lpad(h.county_fips, 5, '0'::text) = lpad(ps.aha_county_fips::text, 5, '0'::text)
             LEFT JOIN vw_tech_enabled_beds_by_county_adjpd b_1 ON lpad(h.county_fips, 5, '0'::text) = lpad(b_1.county_fips, 5, '0'::text)
        ), c1_imputes AS (
         SELECT avg(safe_to_numeric(chr_analytic_chunk_1.premature_death_raw_value::text)) AS avg_premature_death,
            avg(safe_to_numeric(chr_analytic_chunk_1.poor_or_fair_health_raw_value::text)) AS avg_poor_or_fair_health,
            avg(safe_to_numeric(chr_analytic_chunk_1.poor_physical_health_days_raw_value::text)) AS avg_poor_physical_health_days,
            avg(safe_to_numeric(chr_analytic_chunk_1.poor_mental_health_days_raw_value::text)) AS avg_poor_mental_health_days,
            avg(safe_to_numeric(chr_analytic_chunk_1.low_birthweight_raw_value::text)) AS avg_low_birthweight,
            avg(safe_to_numeric(chr_analytic_chunk_1.adult_smoking_raw_value::text)) AS avg_adult_smoking,
            avg(safe_to_numeric(chr_analytic_chunk_1.adult_obesity_raw_value::text)) AS avg_adult_obesity,
            avg(safe_to_numeric(chr_analytic_chunk_1.food_environment_index_raw_value::text)) AS avg_food_environment_index,
            avg(safe_to_numeric(chr_analytic_chunk_1.physical_inactivity_raw_value::text)) AS avg_physical_inactivity,
            avg(safe_to_numeric(chr_analytic_chunk_1.access_to_exercise_opportunities_raw_value::text)) AS avg_access_to_exercise_opportunities,
            avg(safe_to_numeric(chr_analytic_chunk_1.excessive_drinking_raw_value::text)) AS avg_excessive_drinking,
            avg(safe_to_numeric(chr_analytic_chunk_1.alcohol_impaired_driving_deaths_raw_value::text)) AS avg_alcohol_impaired_driving_deaths,
            avg(safe_to_numeric(chr_analytic_chunk_1.sexually_transmitted_infections_raw_value::text)) AS avg_sexually_transmitted_infections,
            avg(safe_to_numeric(chr_analytic_chunk_1.teen_births_raw_value::text)) AS avg_teen_births,
            avg(safe_to_numeric(chr_analytic_chunk_1.uninsured_raw_value::text)) AS avg_uninsured,
            avg(safe_to_numeric(chr_analytic_chunk_1.ratio_of_population_to_primary_care_physicians::text)) AS avg_ratio_of_pop_to_pcp,
            avg(safe_to_numeric(chr_analytic_chunk_1.ratio_of_population_to_dentists::text)) AS avg_ratio_of_pop_to_dentists,
            avg(safe_to_numeric(chr_analytic_chunk_1.ratio_of_population_to_mental_health_providers::text)) AS avg_ratio_of_pop_to_mh,
            avg(safe_to_numeric(chr_analytic_chunk_1.preventable_hospital_stays_raw_value::text)) AS avg_preventable_hospital_stays,
            avg(safe_to_numeric(chr_analytic_chunk_1.mammography_screening_raw_value::text)) AS avg_mammography_screening,
            avg(safe_to_numeric(chr_analytic_chunk_1.flu_vaccinations_raw_value::text)) AS avg_flu_vaccinations,
            avg(safe_to_numeric(chr_analytic_chunk_1.high_school_completion_raw_value::text)) AS avg_high_school_completion,
            avg(safe_to_numeric(chr_analytic_chunk_1.some_college_raw_value::text)) AS avg_some_college,
            avg(safe_to_numeric(chr_analytic_chunk_1.unemployment_raw_value::text)) AS avg_unemployment,
            avg(safe_to_numeric(chr_analytic_chunk_1.children_in_poverty_raw_value::text)) AS avg_children_in_poverty
           FROM chr_analytic_chunk_1
        ), c2_imputes AS (
         SELECT avg(safe_to_numeric(chr_analytic_chunk_2.air_pollution_particulate_matter_raw_value::text)) AS avg_air_pollution,
            avg(safe_to_numeric(chr_analytic_chunk_2.drinking_water_violations_raw_value::text)) AS avg_drinking_water_violations,
            avg(safe_to_numeric(chr_analytic_chunk_2.severe_housing_problems_raw_value::text)) AS avg_severe_housing_problems,
            avg(safe_to_numeric(chr_analytic_chunk_2.driving_alone_to_work_raw_value::text)) AS avg_driving_alone_to_work,
            avg(safe_to_numeric(chr_analytic_chunk_2.long_commute_driving_alone_raw_value::text)) AS avg_long_commute_driving_alone,
            avg(safe_to_numeric(chr_analytic_chunk_2.income_inequality_raw_value::text)) AS avg_income_inequality,
            avg(safe_to_numeric(chr_analytic_chunk_2.children_in_single_parent_households_raw_value::text)) AS avg_children_in_single_parent_households,
            avg(safe_to_numeric(chr_analytic_chunk_2.social_associations_raw_value::text)) AS avg_social_associations,
            avg(safe_to_numeric(chr_analytic_chunk_2.injury_deaths_raw_value::text)) AS avg_injury_deaths
           FROM chr_analytic_chunk_2
        ), new_scores AS (
         SELECT COALESCE(c1._5_digit_fips, c2._5_digit_fips) AS county_fips,
            COALESCE(safe_to_numeric(c1.premature_death_raw_value::text), c1i.avg_premature_death) * 0.50 + COALESCE(safe_to_numeric(c1.poor_or_fair_health_raw_value::text), c1i.avg_poor_or_fair_health) * 0.10 + COALESCE(safe_to_numeric(c1.poor_physical_health_days_raw_value::text), c1i.avg_poor_physical_health_days) * 0.10 + COALESCE(safe_to_numeric(c1.poor_mental_health_days_raw_value::text), c1i.avg_poor_mental_health_days) * 0.10 + COALESCE(safe_to_numeric(c1.low_birthweight_raw_value::text), c1i.avg_low_birthweight) * 0.20 AS health_outcomes_score,
            COALESCE(safe_to_numeric(c1.adult_smoking_raw_value::text), c1i.avg_adult_smoking) * 0.33 + COALESCE(safe_to_numeric(c1.adult_obesity_raw_value::text), c1i.avg_adult_obesity) * 0.17 + COALESCE(safe_to_numeric(c1.food_environment_index_raw_value::text), c1i.avg_food_environment_index) * 0.07 + COALESCE(safe_to_numeric(c1.physical_inactivity_raw_value::text), c1i.avg_physical_inactivity) * 0.07 + COALESCE(safe_to_numeric(c1.access_to_exercise_opportunities_raw_value::text), c1i.avg_access_to_exercise_opportunities) * 0.03 + COALESCE(safe_to_numeric(c1.excessive_drinking_raw_value::text), c1i.avg_excessive_drinking) * 0.08 + COALESCE(safe_to_numeric(c1.alcohol_impaired_driving_deaths_raw_value::text), c1i.avg_alcohol_impaired_driving_deaths) * 0.08 + COALESCE(safe_to_numeric(c1.sexually_transmitted_infections_raw_value::text), c1i.avg_sexually_transmitted_infections) * 0.08 + COALESCE(safe_to_numeric(c1.teen_births_raw_value::text), c1i.avg_teen_births) * 0.08 AS health_behaviors_score,
            COALESCE(safe_to_numeric(c1.uninsured_raw_value::text), c1i.avg_uninsured) * 0.25 + COALESCE(safe_to_numeric(c1.ratio_of_population_to_primary_care_physicians::text), c1i.avg_ratio_of_pop_to_pcp) * 0.15 + COALESCE(safe_to_numeric(c1.ratio_of_population_to_dentists::text), c1i.avg_ratio_of_pop_to_dentists) * 0.05 + COALESCE(safe_to_numeric(c1.ratio_of_population_to_mental_health_providers::text), c1i.avg_ratio_of_pop_to_mh) * 0.05 + COALESCE(safe_to_numeric(c1.preventable_hospital_stays_raw_value::text), c1i.avg_preventable_hospital_stays) * 0.25 + COALESCE(safe_to_numeric(c1.mammography_screening_raw_value::text), c1i.avg_mammography_screening) * 0.13 + COALESCE(safe_to_numeric(c1.flu_vaccinations_raw_value::text), c1i.avg_flu_vaccinations) * 0.13 AS clinical_care_score,
            COALESCE(safe_to_numeric(c2.air_pollution_particulate_matter_raw_value::text), c2i.avg_air_pollution) * 0.25 + COALESCE(safe_to_numeric(c2.drinking_water_violations_raw_value::text), c2i.avg_drinking_water_violations) * 0.25 + COALESCE(safe_to_numeric(c2.severe_housing_problems_raw_value::text), c2i.avg_severe_housing_problems) * 0.20 + COALESCE(safe_to_numeric(c2.driving_alone_to_work_raw_value::text), c2i.avg_driving_alone_to_work) * 0.20 + COALESCE(safe_to_numeric(c2.long_commute_driving_alone_raw_value::text), c2i.avg_long_commute_driving_alone) * 0.10 AS physical_environment_score,
            COALESCE(safe_to_numeric(c1.high_school_completion_raw_value::text), c1i.avg_high_school_completion) * 0.13 + COALESCE(safe_to_numeric(c1.some_college_raw_value::text), c1i.avg_some_college) * 0.13 + COALESCE(safe_to_numeric(c1.unemployment_raw_value::text), c1i.avg_unemployment) * 0.25 + COALESCE(safe_to_numeric(c1.children_in_poverty_raw_value::text), c1i.avg_children_in_poverty) * 0.19 + COALESCE(safe_to_numeric(c2.income_inequality_raw_value::text), c2i.avg_income_inequality) * 0.06 + COALESCE(safe_to_numeric(c2.children_in_single_parent_households_raw_value::text), c2i.avg_children_in_single_parent_households) * 0.06 + COALESCE(safe_to_numeric(c2.social_associations_raw_value::text), c2i.avg_social_associations) * 0.06 + COALESCE(safe_to_numeric(c2.injury_deaths_raw_value::text), c2i.avg_injury_deaths) * 0.13 AS social_economic_factors_score
           FROM chr_analytic_chunk_1 c1
             FULL JOIN chr_analytic_chunk_2 c2 ON c1._5_digit_fips::text = c2._5_digit_fips::text
             CROSS JOIN c1_imputes c1i
             CROSS JOIN c2_imputes c2i
        ), final_view AS (
         SELECT ns.county_fips::text AS county_fips,
            COALESCE(
                CASE
                    WHEN sum(hd.total_adjpd) > 0::numeric THEN sum(hd.ai_adoption_score * hd.total_adjpd) / sum(hd.total_adjpd)
                    ELSE NULL::numeric
                END, 0::numeric) AS weighted_ai_adoption_score,
            COALESCE(
                CASE
                    WHEN sum(hd.total_adjpd) > 0::numeric THEN sum(hd.robotics_adoption_score * hd.total_adjpd) / sum(hd.total_adjpd)
                    ELSE NULL::numeric
                END, 0::numeric) AS weighted_robotics_adoption_score,
                CASE
                    WHEN count(hd.county_fips) > 0 THEN avg(hd.patient_services_margin)
                    ELSE NULL::numeric
                END AS avg_patient_services_margin,
            COALESCE(max(m.exp_active_2019::integer), 0) AS medicaid_expansion_active,
            ns.physical_environment_score,
            ns.health_behaviors_score,
            ns.clinical_care_score,
            ns.health_outcomes_score,
            ns.social_economic_factors_score,
            uscounties.population,
            ac.census_division,
                CASE
                    WHEN count(hd.county_fips) = 0 THEN 1
                    WHEN count(hd.county_fips) > 0 AND sum(
                    CASE
                        WHEN hd.ai_adoption_score > 0::numeric OR hd.robotics_adoption_score > 0::numeric THEN 1
                        ELSE 0
                    END) = 0 THEN 2
                    WHEN sum(
                    CASE
                        WHEN hd.ai_adoption_score > 0::numeric AND hd.robotics_adoption_score > 0::numeric THEN 1
                        ELSE 0
                    END) > 1 THEN 6
                    WHEN sum(
                    CASE
                        WHEN hd.ai_adoption_score > 0::numeric AND hd.robotics_adoption_score > 0::numeric THEN 1
                        ELSE 0
                    END) = 1 THEN 5
                    WHEN sum(
                    CASE
                        WHEN hd.ai_adoption_score > 0::numeric THEN 1
                        ELSE 0
                    END) > 0 THEN 4
                    WHEN sum(
                    CASE
                        WHEN hd.robotics_adoption_score > 0::numeric THEN 1
                        ELSE 0
                    END) > 0 THEN 3
                    ELSE 2
                END AS county_category
           FROM new_scores ns
             LEFT JOIN hospital_data hd ON ns.county_fips::text = hd.county_fips
             LEFT JOIN vw_medicaid_expansion m ON ns.county_fips::text = lpad(m.county_fips_code, 5, '0'::text)
             LEFT JOIN uscounties ON ns.county_fips::text = lpad(uscounties.county_fips::text, 5, '0'::text)
             LEFT JOIN aha_appendix_c_census_divisions ac ON lower(uscounties.state_name::text) = lower(ac.state_desc::text)
          WHERE ns.county_fips::text !~~ '%000'::text
          GROUP BY ns.county_fips, ns.physical_environment_score, ns.health_behaviors_score, ns.clinical_care_score, ns.health_outcomes_score, ns.social_economic_factors_score, uscounties.population, ac.census_division
          ORDER BY ns.county_fips
        )
 SELECT county_fips,
    weighted_ai_adoption_score,
    weighted_robotics_adoption_score,
    avg_patient_services_margin,
    medicaid_expansion_active,
    physical_environment_score,
    health_behaviors_score,
    clinical_care_score,
    health_outcomes_score,
    social_economic_factors_score,
    population,
    census_division,
    county_category
   FROM final_view;
