from __future__ import annotations

from conftest import assert_close, load_table, one_row


def test_primary_county_headline_estimates_match_manuscript() -> None:
    df = load_table("primary_county_crossfit_summary.csv")

    mo14_dv21 = one_row(df, Label="MO14\u2192DV21")
    assert int(mo14_dv21["N"]) == 2898
    assert int(mo14_dv21["N_Treated"]) == 414
    assert_close(mo14_dv21["Crossfit_AIPW_ATE"], -975.821025, tol=1e-5)
    assert_close(mo14_dv21["Crossfit_AIPW_CI_Lower"], -1378.766028, tol=1e-5)
    assert_close(mo14_dv21["Crossfit_AIPW_CI_Upper"], -572.876021, tol=1e-5)
    assert float(mo14_dv21["Crossfit_AIPW_p"]) < 0.001

    mo14_ct6 = one_row(df, Label="MO14\u2192CT6")
    assert int(mo14_ct6["N"]) == 2896
    assert_close(mo14_ct6["Crossfit_AIPW_ATE"], -25.492859, tol=1e-5)
    assert_close(mo14_ct6["Crossfit_Relative_Change_Pct"], -9.900592, tol=1e-5)
    assert float(mo14_ct6["Crossfit_AIPW_p"]) < 0.001

    mo21_ct6 = one_row(df, Label="MO21\u2192CT6")
    assert_close(mo21_ct6["Crossfit_AIPW_ATE"], -25.800115, tol=1e-5)
    assert float(mo21_ct6["Crossfit_AIPW_p"]) < 0.01


def test_mo14_dv21_estimator_triangulation_is_directionally_consistent() -> None:
    df = load_table("method_comparison_mo14_dv21.csv")
    assert len(df) == 6
    assert (df["ATE"] < 0).all()

    non_cf = one_row(df, Method="AIPW (Augmented Inverse Probability Weighting)")
    assert_close(non_cf["ATE"], -457.513054, tol=1e-5)
    assert float(non_cf["p_value"]) < 0.01

    tmle = one_row(df, Method="TMLE (Targeted Maximum Likelihood Estimation)")
    assert_close(tmle["ATE"], -412.698602, tol=1e-5)
    assert float(tmle["p_value"]) < 0.01

    spatial = one_row(df, Method="Spatial Block Bootstrap (AIPW with State Clustering)")
    assert_close(spatial["ATE"], -498.357323, tol=1e-5)
    assert float(spatial["p_value"]) <= 0.001


def test_overlap_weighting_preserves_dv21_but_not_ct6() -> None:
    df = load_table("overlap_weighting_ato_comparison.csv")

    dv21 = one_row(df, Label="MO14\u2192DV21 (YPLL 2020-22)")
    assert_close(dv21["ATO_ATE"], -201.569538, tol=1e-5)
    assert round(float(dv21["ATO_ATE"]), 1) == -201.6
    assert float(dv21["ATO_CI_Upper"]) < 0
    assert float(dv21["ATO_p"]) < 0.01

    ct6 = one_row(df, Label="MO14\u2192CT6 (Hospital Deaths 2023)")
    assert_close(ct6["ATO_ATE"], -2.802004, tol=1e-5)
    assert float(ct6["ATO_CI_Lower"]) < 0 < float(ct6["ATO_CI_Upper"])
    assert float(ct6["ATO_p"]) > 0.05


def test_overlap_weighting_table_reconciles_with_primary_summary() -> None:
    primary = load_table("primary_county_crossfit_summary.csv")
    overlap = load_table("overlap_weighting_ato_comparison.csv")
    label_map = {
        "MO14\u2192DV21": "MO14\u2192DV21 (YPLL 2020-22)",
        "MO14\u2192CT5": "MO14\u2192CT5 (YPLL 2023 Mid)",
        "MO14\u2192CT6": "MO14\u2192CT6 (Hospital Deaths 2023)",
        "MO21\u2192DV21": "MO21\u2192DV21 (YPLL 2020-22)",
        "MO21\u2192CT5": "MO21\u2192CT5 (YPLL 2023 Mid)",
        "MO21\u2192CT6": "MO21\u2192CT6 (Hospital Deaths 2023)",
    }
    for primary_label, overlap_label in label_map.items():
        p = one_row(primary, Label=primary_label)
        o = one_row(overlap, Label=overlap_label)
        for col in [
            "N",
            "N_Treated",
            "N_Control",
            "Control_Mean",
            "ATO_ATE",
            "ATO_CI_Lower",
            "ATO_CI_Upper",
            "ATO_p",
            "ATO_ESS",
        ]:
            assert_close(o[col], p[col], tol=1e-5)


def test_hospital_confirmatory_headlines_match_main_text() -> None:
    df = load_table("main_hospital_confirmatory_results.csv")

    sep1 = one_row(df, result_id="mo11_sep1")
    assert int(sep1["n"]) == 6166
    assert_close(sep1["estimate"], 2.24, tol=1e-6)
    assert_close(sep1["ci_lower"], 1.03, tol=1e-6)
    assert_close(sep1["ci_upper"], 3.44, tol=1e-6)
    assert_close(sep1["relative_change_pct"], 3.9, tol=1e-6)

    pneumonia = one_row(df, result_id="mo14_pneumonia")
    assert int(pneumonia["n"]) == 6166
    assert_close(pneumonia["estimate"], -0.87, tol=1e-6)
    assert_close(pneumonia["ci_lower"], -1.35, tol=1e-6)
    assert_close(pneumonia["ci_upper"], -0.40, tol=1e-6)
    assert_close(pneumonia["relative_change_pct"], -5.4, tol=1e-6)


def test_access_inequality_headlines_match_main_text_and_si() -> None:
    df = load_table("access_inequality_key_results.csv")

    ai_coverage = one_row(df, result_id="coverage_ai_30min_2023")
    robotics_coverage = one_row(df, result_id="coverage_robotics_30min_2023")
    assert_close(ai_coverage["value"], 65.8, tol=1e-6)
    assert_close(robotics_coverage["value"], 79.5, tol=1e-6)

    ai_penalty = one_row(df, result_id="p90_p10_ratio_ai_2023")
    assert_close(ai_penalty["value"], 32.3, tol=1e-6)
    assert_close(one_row(df, result_id="p90_distance_ai_2023")["value"], 61.3, tol=1e-6)

    assert_close(one_row(df, result_id="gini_all_hospitals_2023")["value"], 0.829, tol=1e-6)
    assert_close(one_row(df, result_id="gini_robotics_2023")["value"], 0.776, tol=1e-6)
    assert_close(one_row(df, result_id="gini_ai_2023")["value"], 0.740, tol=1e-6)

    assert_close(one_row(df, result_id="ai_enabled_hospitals_2022")["value"], 1119, tol=1e-6)
    assert_close(one_row(df, result_id="ai_enabled_hospitals_2024")["value"], 1743, tol=1e-6)
    assert_close(one_row(df, result_id="ai_enabled_hospitals_growth_2022_2024")["value"], 56, tol=1e-6)
    assert_close(one_row(df, result_id="ai_gini_2022")["value"], 0.739, tol=1e-6)
    assert_close(one_row(df, result_id="ai_gini_2024")["value"], 0.767, tol=1e-6)


def test_pre_exposure_checks_support_sep1_caution_on_pneumonia() -> None:
    df = load_table("revision_diagnostics/table_s3_pre_exposure_balance_results.csv")

    sep1_incident = one_row(
        df,
        comparison="Incident 2023 adopter vs stable nonadopter",
        exposure="wfaiss",
        measure="SEP_1",
    )
    assert_close(sep1_incident["difference"], 1.228, tol=1e-6)
    assert float(sep1_incident["ci_low"]) < 0 < float(sep1_incident["ci_high"])
    assert float(sep1_incident["p_value"]) > 0.40

    sep1_change = one_row(
        df,
        comparison="Incident 2023 adopter vs stable nonadopter",
        exposure="wfaiss",
        measure="SEP_1_CHANGE",
    )
    assert float(sep1_change["p_value"]) > 0.40

    pn_incident = one_row(
        df,
        comparison="Incident 2023 adopter vs stable nonadopter",
        exposure="wfaiart",
        measure="MORT_30_PN",
    )
    assert_close(pn_incident["difference"], -0.4534235325, tol=1e-6)
    assert float(pn_incident["ci_high"]) < 0
    assert float(pn_incident["p_value"]) < 0.05


def test_system_membership_diagnostic_matches_response_to_reviewer3() -> None:
    balance = load_table("revision_diagnostics/table_s27_system_membership_balance.csv")
    mo14 = one_row(balance, analysis_level="Hospital", measure="MO14 routine-task AI")
    assert_close(mo14["system_member_mean_pct"], 34.4378698225, tol=1e-6)
    assert_close(mo14["non_system_mean_pct"], 9.9836333879, tol=1e-6)
    assert float(mo14["smd"]) > 0.60

    county_share = one_row(
        balance,
        analysis_level="County",
        measure="System-member share of AHA adjusted patient days",
    )
    assert_close(county_share["system_member_mean_pct"], 82.8295796851, tol=1e-6)
    assert_close(county_share["non_system_mean_pct"], 43.0279334967, tol=1e-6)
    assert float(county_share["smd"]) > 1.0

    aipw = load_table("revision_diagnostics/table_s27_system_membership_aipw_sensitivity.csv")
    base = one_row(aipw, outcome="dv21_ypll", specification="Base covariates only")
    adjusted = one_row(
        aipw,
        outcome="dv21_ypll",
        specification="Base covariates + system-member capacity share",
    )
    assert_close(base["estimate"], -438.2462856788, tol=1e-6)
    assert_close(adjusted["estimate"], -355.0981282649, tol=1e-6)
    assert float(adjusted["ci_upper"]) < 0


def test_organizational_capacity_diagnostic_matches_response_to_reviewer3() -> None:
    balance = load_table("revision_diagnostics/table_s28_organizational_capacity_balance.csv")
    county_mo14 = balance[
        (balance["level"] == "County") & (balance["exposure"] == "County any MO14 access")
    ]
    assert not county_mo14.empty

    rn_bed = one_row(county_mo14, measure="RN FTE per bed")
    assert_close(rn_bed["exposed_mean"], 2.1127307675, tol=1e-6)
    assert_close(rn_bed["unexposed_mean"], 1.8026925964, tol=1e-6)
    assert float(rn_bed["smd"]) > 0.30

    capex = one_row(county_mo14, measure="log CAPEX per adjusted patient day")
    assert_close(capex["exposed_mean"], 4.5145396386, tol=1e-6)
    assert_close(capex["unexposed_mean"], 3.9841060492, tol=1e-6)
    assert float(capex["smd"]) > 0.40

    aipw = load_table("revision_diagnostics/table_s28_organizational_capacity_aipw_sensitivity.csv")
    dv21_base = one_row(
        aipw,
        level="County",
        contrast="mo14_any -> dv21_ypll",
        specification="Base covariates only",
    )
    dv21_adjusted = one_row(
        aipw,
        level="County",
        contrast="mo14_any -> dv21_ypll",
        specification="Base + RN staffing/CAPEX/HCRIS/HHI proxies",
    )
    assert_close(dv21_base["estimate"], -438.2462856788, tol=1e-6)
    assert_close(dv21_adjusted["estimate"], -279.6318147434, tol=1e-6)
    assert float(dv21_adjusted["ci_upper"]) < 0

    ct6_adjusted = one_row(
        aipw,
        level="County",
        contrast="mo14_any -> ct6_hospital_deaths",
        specification="Base + RN staffing/CAPEX/HCRIS/HHI proxies",
    )
    assert_close(ct6_adjusted["estimate"], -0.0620966971, tol=1e-6)
    assert float(ct6_adjusted["ci_lower"]) < 0 < float(ct6_adjusted["ci_upper"])


def test_exposure_misclassification_preserves_mo14_dv21_direction_only() -> None:
    df = load_table("county_treatment_misclassification_sensitivity.csv")
    mo14_dv21 = df[df["Label"].eq("MO14\u2192DV21")]
    assert len(mo14_dv21) == 6
    assert (mo14_dv21["ATE_Median"] < 0).all()
    assert mo14_dv21["ATE_Median"].between(-600, -80).all()

    mo14_ct6 = df[df["Label"].eq("MO14\u2192CT6")]
    assert (mo14_ct6["ATE_Median"] > 0).any()
    assert (mo14_ct6["ATE_Median"] < 0).any()
