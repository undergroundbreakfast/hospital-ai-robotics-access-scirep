#!/usr/bin/env python3
"""Build post-hoc organizational-capacity diagnostics for the revision.

This analysis addresses Reviewer 3's residual-confounding concern using
variables already available in AHA/CMS-linked files:

* AHA RN staffing-intensity proxies, not shift-level bedside assignment ratios.
* AHA total capital-expenditure intensity, not health-IT-specific CAPEX.
* Contemporary AHA market concentration measures based on system-level shares
  of adjusted patient days within CBSAs and counties.
* CMS HCRIS-derived operating margin, total margin, and days-cash-on-hand
  liquidity proxies keyed by CMS Certification Number.

The estimators are intentionally conservative linear-nuisance AIPW diagnostics.
They are not replacements for the pre-specified primary models.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
import psycopg


def connect() -> psycopg.Connection:
    password = os.getenv("POSTGRESQL_KEY") or os.getenv("PGPASSWORD")
    if not password:
        raise SystemExit("Set POSTGRESQL_KEY or PGPASSWORD before running this script.")
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "localhost")),
        port=int(os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432"))),
        dbname=os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "Research_TEST")),
        user=os.getenv("POSTGRES_USER", os.getenv("PGUSER", "postgres")),
        password=password,
    )


def read_sql(conn: psycopg.Connection, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def num_sql(col: str) -> str:
    return (
        f"case when nullif(regexp_replace({col}::text, '[^0-9.\\-]', '', 'g'), '') is not null "
        f"then nullif(regexp_replace({col}::text, '[^0-9.\\-]', '', 'g'), '')::numeric end"
    )


def hhi_ctes() -> str:
    adjpd = num_sql("adjpd")
    return f"""
    aha_market_base as (
        select
            id::text as aha_id,
            lpad(fcounty::text, 5, '0') as county_fips,
            nullif(cbsacode::text, '') as cbsa_code,
            coalesce(nullif(sysid::text, ''), 'HOSP_' || id::text) as firm_id,
            {adjpd} as adjpd
        from public.aha_fy2024
        where {adjpd} is not null and {adjpd} > 0
    ),
    market_firms as (
        select 'CBSA'::text as market_type, cbsa_code as market_id, firm_id, sum(adjpd) as firm_adjpd
        from aha_market_base
        where cbsa_code is not null
        group by cbsa_code, firm_id
        union all
        select 'County'::text as market_type, county_fips as market_id, firm_id, sum(adjpd) as firm_adjpd
        from aha_market_base
        where county_fips is not null
        group by county_fips, firm_id
    ),
    market_totals as (
        select market_type, market_id, count(*) as firm_count, sum(firm_adjpd) as total_adjpd
        from market_firms
        group by market_type, market_id
    ),
    market_hhi as (
        select
            mf.market_type,
            mf.market_id,
            mt.firm_count,
            mt.total_adjpd,
            sum(power(mf.firm_adjpd / nullif(mt.total_adjpd, 0), 2)) * 10000 as hhi
        from market_firms mf
        join market_totals mt
          on mt.market_type = mf.market_type
         and mt.market_id = mf.market_id
        group by mf.market_type, mf.market_id, mt.firm_count, mt.total_adjpd
    )
    """


def hospital_sql() -> str:
    ftern = num_sql("a.ftern")
    crnfte = num_sql("a.crnfte")
    vrn = num_sql("a.vrn")
    adjpd = num_sql("a.adjpd")
    bdtot = num_sql("a.bdtot")
    ceamt = num_sql("a.ceamt")
    gfeet = num_sql("a.gfeet")
    v_adjpd = num_sql("v.adjpd")
    return f"""
    with
    {hhi_ctes()},
    hospital_base as (
        select
            v.aha_id,
            v.cms_facility_id,
            substring(v.aha_county_fips from 1 for 2) as state_fips,
            v.state,
            v.census_division,
            v.hospital_ownership,
            v.bsc,
            case when v.wfaiss = '1' then 1 when v.wfaiss = '0' then 0 end as mo11_staff_scheduling,
            case when v.wfaiart = '1' then 1 when v.wfaiart = '0' then 0 end as mo14_routine_tasks,
            case when v.robohos = '1' then 1 when v.robohos = '0' then 0 end as mo21_robotics,
            {num_sql("v.sep_1")} as sep1_2023,
            {num_sql("v.mort_30_pn_score")} as pn_mort_2023,
            {num_sql("p.sep_1_2019")} as sep1_2019,
            {num_sql("p.mort_30_pn_2019")} as pn_mort_2019,
            v.social_economic_factors_score::numeric as social_economic,
            v.health_behaviors_score::numeric as health_behaviors,
            v.physical_environment_score::numeric as physical_environment,
            v.medicaid_expansion_active::numeric as medicaid_expansion,
            {v_adjpd} as model_adjpd,
            {bdtot} as model_beds,
            {ftern} as rn_fte,
            {crnfte} as contract_rn_fte,
            {vrn} as rn_vacancies,
            {adjpd} as aha_adjpd,
            {bdtot} as aha_beds,
            {ceamt} as capex_total,
            {gfeet} as square_feet,
            cbsa.hhi as cbsa_hhi,
            county.hhi as county_hhi,
            hcris.operating_margin as hcris_operating_margin,
            hcris.total_margin as hcris_total_margin,
            hcris.days_cash_on_hand as hcris_days_cash_on_hand
        from public.vw_hospital_level_aipw v
        left join public.hospital_level_placebo_2019 p
          on p.facility_id::text = v.cms_facility_id::text
        left join public.aha_survey_data a
          on a.id::text = v.aha_id::text
        left join public.aha_fy2024 a24
          on a24.id::text = v.aha_id::text
        left join market_hhi cbsa
          on cbsa.market_type = 'CBSA'
         and cbsa.market_id = nullif(a24.cbsacode::text, '')
        left join market_hhi county
          on county.market_type = 'County'
         and county.market_id = lpad(a24.fcounty::text, 5, '0')
        left join public.cms_hcris_2023_financial_capacity hcris
          on hcris.provider_ccn = lpad(v.cms_facility_id::text, 6, '0')
    )
    select
        *,
        case when aha_adjpd > 0 then rn_fte / aha_adjpd * 1000 end as rn_fte_per_1000_adjpd,
        case when aha_beds > 0 then rn_fte / aha_beds end as rn_fte_per_bed,
        case when (rn_fte + rn_vacancies) > 0 then rn_vacancies / (rn_fte + rn_vacancies) end as rn_vacancy_rate,
        case when (rn_fte + contract_rn_fte) > 0 then contract_rn_fte / (rn_fte + contract_rn_fte) end as contract_rn_share,
        case when aha_adjpd > 0 then capex_total / aha_adjpd end as capex_per_adjpd,
        case when square_feet > 0 then capex_total / square_feet end as capex_per_sqft,
        case when model_adjpd > 0 then ln(model_adjpd) end as ln_adjpd,
        case when model_beds > 0 then ln(model_beds) end as ln_beds
    from hospital_base
    """


def county_sql() -> str:
    ftern = num_sql("a.ftern")
    crnfte = num_sql("a.crnfte")
    vrn = num_sql("a.vrn")
    adjpd = num_sql("a.adjpd")
    bdtot = num_sql("a.bdtot")
    ceamt = num_sql("a.ceamt")
    gfeet = num_sql("a.gfeet")
    return f"""
    with
    {hhi_ctes()},
    hospital_proxies as (
        select
            lpad(a.fcounty::text, 5, '0') as county_fips,
            {ftern} as rn_fte,
            {crnfte} as contract_rn_fte,
            {vrn} as rn_vacancies,
            {adjpd} as adjpd,
            {bdtot} as beds,
            {ceamt} as capex_total,
            {gfeet} as square_feet,
            cbsa.hhi as cbsa_hhi,
            hcris.operating_margin as hcris_operating_margin,
            hcris.total_margin as hcris_total_margin,
            hcris.days_cash_on_hand as hcris_days_cash_on_hand
        from public.aha_survey_data a
        left join public.aha_fy2024 a24
          on a24.id::text = a.id::text
        left join market_hhi cbsa
          on cbsa.market_type = 'CBSA'
         and cbsa.market_id = nullif(a24.cbsacode::text, '')
        left join public.cms_hcris_2023_financial_capacity hcris
          on hcris.provider_ccn = lpad(a24.mcrnum::text, 6, '0')
        where a.fcounty is not null
    ),
    county_proxy as (
        select
            county_fips,
            count(*) as aha_hospitals_with_proxy_frame,
            sum(rn_fte) as rn_fte_sum,
            sum(rn_vacancies) as rn_vacancies_sum,
            sum(adjpd) as adjpd_sum,
            sum(beds) as beds_sum,
            sum(capex_total) as capex_sum,
            sum(square_feet) as square_feet_sum,
            case when sum(adjpd) > 0 then sum(rn_fte) / sum(adjpd) * 1000 end as rn_fte_per_1000_adjpd,
            case when sum(beds) > 0 then sum(rn_fte) / sum(beds) end as rn_fte_per_bed,
            case when (sum(rn_fte) + sum(rn_vacancies)) > 0 then sum(rn_vacancies) / (sum(rn_fte) + sum(rn_vacancies)) end as rn_vacancy_rate,
            case when (sum(rn_fte) + sum(contract_rn_fte)) > 0 then sum(contract_rn_fte) / (sum(rn_fte) + sum(contract_rn_fte)) end as contract_rn_share,
            case when sum(adjpd) > 0 then sum(capex_total) / sum(adjpd) end as capex_per_adjpd,
            case when sum(square_feet) > 0 then sum(capex_total) / sum(square_feet) end as capex_per_sqft,
            case when sum(adjpd) > 0 then sum(coalesce(cbsa_hhi, 0) * adjpd) / sum(adjpd) end as adjpd_weighted_cbsa_hhi,
            case when sum(case when hcris_operating_margin is not null then adjpd end) > 0
                 then sum(hcris_operating_margin * adjpd)
                    / sum(case when hcris_operating_margin is not null then adjpd end)
            end as hcris_operating_margin,
            case when sum(case when hcris_total_margin is not null then adjpd end) > 0
                 then sum(hcris_total_margin * adjpd)
                    / sum(case when hcris_total_margin is not null then adjpd end)
            end as hcris_total_margin,
            case when sum(case when hcris_days_cash_on_hand is not null then adjpd end) > 0
                 then sum(hcris_days_cash_on_hand * adjpd)
                    / sum(case when hcris_days_cash_on_hand is not null then adjpd end)
            end as hcris_days_cash_on_hand
        from hospital_proxies
        group by county_fips
    )
    select
        c.county_fips,
        substring(c.county_fips from 1 for 2) as state_fips,
        c.census_division,
        {num_sql("c.population")} as population,
        {num_sql("c.pl1_ypll_rate_2019")} as ypll_2019,
        c.iv3_health_behaviors_score::numeric as health_behaviors,
        c.iv4_social_economic_factors_score::numeric as social_economic,
        c.iv2_physical_environment_score::numeric as physical_environment,
        c.iv1_medicaid_expansion_active::numeric as medicaid_expansion,
        c.dv21_premature_death_ypll_rate::numeric as dv21_ypll,
        c.ct6_hospital_deaths_age_adj_2023::numeric as ct6_hospital_deaths,
        c.mo14_ai_automate_routine_tasks_pct::numeric as mo14_pct,
        c.mo21_robotics_in_hospital_pct::numeric as mo21_pct,
        cp.rn_fte_per_1000_adjpd,
        cp.rn_fte_per_bed,
        cp.rn_vacancy_rate,
        cp.contract_rn_share,
        cp.capex_per_adjpd,
        cp.capex_per_sqft,
        cp.adjpd_weighted_cbsa_hhi,
        cp.hcris_operating_margin,
        cp.hcris_total_margin,
        cp.hcris_days_cash_on_hand,
        chhi.hhi as county_hhi
    from public.vw_county_file_export_wide c
    left join county_proxy cp
      on cp.county_fips = c.county_fips
    left join market_hhi chhi
      on chhi.market_type = 'County'
     and chhi.market_id = c.county_fips
    """


def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["capex_per_adjpd", "capex_per_sqft"]:
        out[f"ln_{col}"] = np.log1p(pd.to_numeric(out[col], errors="coerce").astype(float))
    for col in ["hcris_operating_margin", "hcris_total_margin", "hcris_days_cash_on_hand"]:
        if col in out.columns:
            x = pd.to_numeric(out[col], errors="coerce").astype(float)
            if x.notna().sum() >= 20:
                lo = x.quantile(0.01)
                hi = x.quantile(0.99)
                out[f"{col}_w"] = x.clip(lower=lo, upper=hi)
            else:
                out[f"{col}_w"] = x
    if "hcris_days_cash_on_hand_w" in out.columns:
        out["ln_hcris_days_cash_on_hand"] = np.log1p(
            pd.to_numeric(out["hcris_days_cash_on_hand_w"], errors="coerce").astype(float).clip(lower=0)
        )
    for col in ["cbsa_hhi", "county_hhi", "adjpd_weighted_cbsa_hhi"]:
        if col in out.columns:
            out[f"{col}_per_1000"] = pd.to_numeric(out[col], errors="coerce").astype(float) / 1000.0
    for col in [
        "rn_fte_per_1000_adjpd",
        "rn_fte_per_bed",
        "rn_vacancy_rate",
        "contract_rn_share",
        "ln_capex_per_adjpd",
        "ln_capex_per_sqft",
        "cbsa_hhi_per_1000",
        "county_hhi_per_1000",
        "adjpd_weighted_cbsa_hhi_per_1000",
        "hcris_operating_margin_w",
        "hcris_total_margin_w",
        "ln_hcris_days_cash_on_hand",
    ]:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(float)
    return out


def smd_cont(x1: pd.Series, x0: pd.Series) -> float:
    sd1 = x1.std(ddof=1)
    sd0 = x0.std(ddof=1)
    pooled = math.sqrt((sd1**2 + sd0**2) / 2)
    return float((x1.mean() - x0.mean()) / pooled) if pooled else float("nan")


def balance_rows(df: pd.DataFrame, level: str) -> pd.DataFrame:
    exposures = [
        ("mo11_staff_scheduling", "MO11 staff-scheduling AI"),
        ("mo14_routine_tasks", "MO14 routine-task AI"),
        ("mo21_robotics", "MO21 in-hospital robotics"),
    ] if level == "Hospital" else [
        ("mo14_any", "County any MO14 access"),
        ("mo21_any", "County any MO21 access"),
    ]
    measures = [
        ("rn_fte_per_1000_adjpd", "RN FTE per 1,000 adjusted patient days"),
        ("rn_fte_per_bed", "RN FTE per bed"),
        ("rn_vacancy_rate", "RN vacancy rate"),
        ("ln_capex_per_adjpd", "log CAPEX per adjusted patient day"),
        ("ln_capex_per_sqft", "log CAPEX per square foot"),
        ("hcris_operating_margin_w", "HCRIS operating margin"),
        ("hcris_total_margin_w", "HCRIS total margin"),
        ("ln_hcris_days_cash_on_hand", "log HCRIS days cash on hand"),
    ]
    if level == "Hospital":
        measures += [
            ("cbsa_hhi_per_1000", "CBSA HHI / 1,000"),
            ("county_hhi_per_1000", "County HHI / 1,000"),
        ]
    else:
        measures += [
            ("adjpd_weighted_cbsa_hhi_per_1000", "County-weighted CBSA HHI / 1,000"),
            ("county_hhi_per_1000", "County HHI / 1,000"),
        ]
    rows = []
    for exposure, exposure_label in exposures:
        for measure, measure_label in measures:
            sub = df.dropna(subset=[exposure, measure]).copy()
            if sub.empty:
                continue
            treated = pd.to_numeric(sub.loc[sub[exposure] == 1, measure], errors="coerce").dropna()
            control = pd.to_numeric(sub.loc[sub[exposure] == 0, measure], errors="coerce").dropna()
            if len(treated) < 5 or len(control) < 5:
                continue
            rows.append(
                {
                    "level": level,
                    "exposure": exposure_label,
                    "measure": measure_label,
                    "n": int(len(treated) + len(control)),
                    "exposed_n": int(len(treated)),
                    "unexposed_n": int(len(control)),
                    "exposed_mean": float(treated.mean()),
                    "unexposed_mean": float(control.mean()),
                    "difference": float(treated.mean() - control.mean()),
                    "smd": smd_cont(treated, control),
                    "exposed_median": float(treated.median()),
                    "unexposed_median": float(control.median()),
                }
            )
    return pd.DataFrame(rows)


def design_matrix(
    df: pd.DataFrame,
    covariates: Iterable[str],
    categorical: Iterable[str],
) -> np.ndarray:
    blocks = []
    for col in covariates:
        x = pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
        mean = np.nanmean(x)
        std = np.nanstd(x)
        x = np.where(np.isfinite(x), x, mean)
        if std > 0:
            x = (x - mean) / std
        else:
            x = x * 0
        blocks.append(x.reshape(-1, 1))
    for col in categorical:
        dummies = pd.get_dummies(df[col].fillna("Missing"), prefix=col, drop_first=True, dtype=float)
        if not dummies.empty:
            blocks.append(dummies.to_numpy(dtype=float))
    return np.column_stack(blocks) if blocks else np.empty((len(df), 0))


def logistic_fit_predict(X: np.ndarray, a: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.zeros(X1.shape[1])
    penalty = np.eye(X1.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(100):
        eta = np.clip(X1 @ beta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        grad = X1.T @ (p - a) + penalty @ beta
        hess = (X1.T * w) @ X1 + penalty
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(X1 @ beta, -35, 35))), 0.01, 0.99)


def ols_predict_potential(X: np.ndarray, a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    XA = np.column_stack([np.ones(X.shape[0]), a, X])
    beta = np.linalg.pinv(XA.T @ XA) @ XA.T @ y
    X1 = np.column_stack([np.ones(X.shape[0]), np.ones(X.shape[0]), X])
    X0 = np.column_stack([np.ones(X.shape[0]), np.zeros(X.shape[0]), X])
    return X1 @ beta, X0 @ beta


def clustered_mean_se(scores: np.ndarray, clusters: pd.Series) -> float:
    psi = scores - np.nanmean(scores)
    cl = clusters.astype(str).to_numpy()
    unique = np.unique(cl)
    n = len(psi)
    g = len(unique)
    if g <= 1:
        return float(np.nanstd(scores, ddof=1) / math.sqrt(n))
    sums = np.array([psi[cl == u].sum() for u in unique])
    var = (g / (g - 1)) * np.sum(sums**2) / (n**2)
    return float(math.sqrt(var))


def aipw(df: pd.DataFrame, level: str, treatment: str, outcome: str, base_covars: list[str],
         capacity_covars: list[str], categorical: list[str], cluster: str,
         include_capacity: bool) -> dict[str, object]:
    covars = list(base_covars)
    if include_capacity:
        covars += capacity_covars
        covars += [f"{c}_missing" for c in capacity_covars if f"{c}_missing" in df.columns]
    needed = [treatment, outcome, cluster] + [c for c in base_covars if not c.endswith("_missing")]
    work = df.dropna(subset=needed).copy()
    a = pd.to_numeric(work[treatment], errors="coerce").astype(float).to_numpy()
    y = pd.to_numeric(work[outcome], errors="coerce").astype(float).to_numpy()
    X = design_matrix(work, covars, categorical)
    e = logistic_fit_predict(X, a)
    mu1, mu0 = ols_predict_potential(X, a, y)
    scores = mu1 - mu0 + (a / e) * (y - mu1) - ((1 - a) / (1 - e)) * (y - mu0)
    estimate = float(np.nanmean(scores))
    se = clustered_mean_se(scores, work[cluster])
    return {
        "level": level,
        "contrast": f"{treatment} -> {outcome}",
        "specification": "Base + RN staffing/CAPEX/HCRIS/HHI proxies" if include_capacity else "Base covariates only",
        "n": int(len(work)),
        "treated_n": int(np.nansum(a)),
        "control_n": int(len(work) - np.nansum(a)),
        "estimate": estimate,
        "se": se,
        "ci_lower": estimate - 1.96 * se,
        "ci_upper": estimate + 1.96 * se,
        "propensity_min": float(np.nanmin(e)),
        "propensity_max": float(np.nanmax(e)),
        "propensity_mean": float(np.nanmean(e)),
    }


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [cols] + [
        ["" if pd.isna(v) else f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        for row in df.to_numpy()
    ]
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(cols))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(cols))) + " |"

    out = [fmt(rows[0]), "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"]
    out.extend(fmt(row) for row in rows[1:])
    return "\n".join(out)


def write_latex(balance: pd.DataFrame, aipw_df: pd.DataFrame) -> None:
    rows_to_show = balance[
        (balance["level"] == "County")
        & (balance["exposure"] == "County any MO14 access")
        & balance["measure"].isin(
            [
                "RN FTE per 1,000 adjusted patient days",
                "RN FTE per bed",
                "RN vacancy rate",
                "log CAPEX per adjusted patient day",
                "HCRIS operating margin",
                "HCRIS total margin",
                "log HCRIS days cash on hand",
                "County-weighted CBSA HHI / 1,000",
                "County HHI / 1,000",
            ]
        )
    ].copy()
    sens_to_show = aipw_df[
        (aipw_df["level"] == "County")
        & aipw_df["contrast"].isin(
            [
                "mo14_any -> dv21_ypll",
                "mo14_any -> ct6_hospital_deaths",
            ]
        )
    ].copy()
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Post-hoc organizational-capacity diagnostics using AHA staffing, capital, market-concentration, and CMS HCRIS financial-capacity proxies}",
        r"\label{tab:organizational_capacity_sensitivity}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Panel & Measure & N & Exposed mean & Unexposed mean & SMD \\",
        r"\midrule",
    ]
    for _, row in rows_to_show.iterrows():
        lines.append(
            f"A & {row['measure']} & {int(row['n']):,} & {row['exposed_mean']:.2f} & "
            f"{row['unexposed_mean']:.2f} & {row['smd']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"Panel & Sensitivity model & N & Estimate & 95\% CI & -- \\",
        ]
    )
    for _, row in sens_to_show.iterrows():
        outcome_label = "DV21 YPLL" if "dv21" in row["contrast"] else "CT6 hospital deaths"
        lines.append(
            f"B & {outcome_label}: {row['specification']} & {int(row['n']):,} & "
            f"{row['estimate']:.1f} & [{row['ci_lower']:.1f}, {row['ci_upper']:.1f}] & -- \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize Notes: Panel A compares counties with any MO14 access to counties with no MO14 access. RN measures are AHA staffing-intensity proxies and should not be interpreted as shift-level bedside assignment ratios. CAPEX is total AHA capital expenditure and is not health-IT-specific CAPEX. HHI is computed from AHA system-level shares of adjusted patient days within contemporary CBSA or county markets. HCRIS operating margin is net income from service to patients divided by net patient revenue; HCRIS total margin is net income divided by net patient revenue plus other income; days cash on hand is cash plus temporary investments divided by report-period daily cash operating expense. HCRIS financial proxies are winsorized at the 1st and 99th percentiles before inclusion in the sensitivity models. Missing organizational-capacity proxies are represented with missingness indicators and mean-imputed in the standardized design matrix. Panel B reports conservative linear-nuisance AIPW diagnostics for MO14 access with and without these organizational-capacity proxies; the base rows are reduced-estimator baseline diagnostics, not the primary cross-fitted Random Forest estimates.",
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )
    (ROOT / "organizational_capacity_sensitivity_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    with connect() as conn:
        hospitals = add_transforms(read_sql(conn, hospital_sql()))
        counties = add_transforms(read_sql(conn, county_sql()))

    counties["mo14_any"] = (pd.to_numeric(counties["mo14_pct"], errors="coerce") > 0).astype(float)
    counties["mo21_any"] = (pd.to_numeric(counties["mo21_pct"], errors="coerce") > 0).astype(float)
    counties["log_population"] = np.log(pd.to_numeric(counties["population"], errors="coerce").astype(float))

    hospital_balance = balance_rows(hospitals, "Hospital")
    county_balance = balance_rows(counties, "County")
    balance = pd.concat([hospital_balance, county_balance], ignore_index=True)

    hospital_capacity_covars = [
        "rn_fte_per_1000_adjpd",
        "rn_fte_per_bed",
        "rn_vacancy_rate",
        "ln_capex_per_adjpd",
        "ln_capex_per_sqft",
        "cbsa_hhi_per_1000",
        "county_hhi_per_1000",
        "hcris_operating_margin_w",
        "hcris_total_margin_w",
        "ln_hcris_days_cash_on_hand",
    ]
    county_capacity_covars = [
        "rn_fte_per_1000_adjpd",
        "rn_fte_per_bed",
        "rn_vacancy_rate",
        "ln_capex_per_adjpd",
        "ln_capex_per_sqft",
        "adjpd_weighted_cbsa_hhi_per_1000",
        "county_hhi_per_1000",
        "hcris_operating_margin_w",
        "hcris_total_margin_w",
        "ln_hcris_days_cash_on_hand",
    ]

    aipw_rows = []
    hospital_specs = [
        ("mo11_staff_scheduling", "sep1_2023", "sep1_2019"),
        ("mo14_routine_tasks", "pn_mort_2023", "pn_mort_2019"),
    ]
    for treatment, outcome, baseline in hospital_specs:
        base_covars = [
            baseline,
            "ln_adjpd",
            "ln_beds",
            "social_economic",
            "health_behaviors",
            "physical_environment",
            "medicaid_expansion",
        ]
        for include_capacity in [False, True]:
            aipw_rows.append(
                aipw(
                    hospitals,
                    "Hospital",
                    treatment,
                    outcome,
                    base_covars,
                    hospital_capacity_covars,
                    ["state", "hospital_ownership", "bsc"],
                    "state_fips",
                    include_capacity,
                )
            )

    county_base_covars = [
        "ypll_2019",
        "health_behaviors",
        "social_economic",
        "physical_environment",
        "medicaid_expansion",
        "log_population",
    ]
    for outcome in ["dv21_ypll", "ct6_hospital_deaths"]:
        for include_capacity in [False, True]:
            aipw_rows.append(
                aipw(
                    counties,
                    "County",
                    "mo14_any",
                    outcome,
                    county_base_covars,
                    county_capacity_covars,
                    ["census_division"],
                    "state_fips",
                    include_capacity,
                )
            )

    aipw_df = pd.DataFrame(aipw_rows)

    balance.to_csv(ROOT / "organizational_capacity_balance.csv", index=False)
    aipw_df.to_csv(ROOT / "organizational_capacity_aipw_sensitivity.csv", index=False)
    hospitals.to_csv(ROOT / "organizational_capacity_hospital_dataset.csv", index=False)
    counties.to_csv(ROOT / "organizational_capacity_county_dataset.csv", index=False)
    write_latex(balance, aipw_df)

    summary = [
        "# Organizational-Capacity Sensitivity",
        "",
        "## Balance Diagnostics",
        markdown_table(balance),
        "",
        "## Linear-Nuisance AIPW Sensitivity",
        markdown_table(aipw_df),
        "",
    ]
    (ROOT / "organizational_capacity_sensitivity_summary.md").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
