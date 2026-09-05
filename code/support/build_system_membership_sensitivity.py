#!/usr/bin/env python3
"""Build post-hoc AHA system-membership sensitivity checks for the revision.

The goal is deliberately narrow: quantify whether AHA health-system membership
is associated with workflow-AI adoption, then test whether adding a county-level
system-member capacity share changes the focal MO14 county association in a
simple linear-nuisance AIPW sensitivity.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirep_config import connect_db, output_directory

ROOT = output_directory("system_membership")

import numpy as np
import pandas as pd
import psycopg


def connect():
    return connect_db()


def read_sql(conn: psycopg.Connection, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def clean_numeric_sql(col: str) -> str:
    return (
        f"case when nullif(regexp_replace({col}::text, '[^0-9.\\-]', '', 'g'), '') is not null "
        f"then nullif(regexp_replace({col}::text, '[^0-9.\\-]', '', 'g'), '')::numeric end"
    )


def hospital_sql() -> str:
    return f"""
    select
        v.aha_id,
        v.cms_facility_id,
        v.state,
        case when a24.mhsmemb = 'Yes' or a24.sysid is not null then 1 else 0 end as system_member,
        a24.sysid as system_id,
        case when v.wfaiss = '1' then 1 when v.wfaiss = '0' then 0 end as mo11_staff_scheduling,
        case when v.wfaiart = '1' then 1 when v.wfaiart = '0' then 0 end as mo14_routine_tasks,
        case when v.robohos = '1' then 1 when v.robohos = '0' then 0 end as mo21_robotics,
        {clean_numeric_sql('v.adjpd')} as adjpd
    from public.vw_hospital_level_aipw v
    left join public.aha_fy2024 a24
      on a24.id::text = v.aha_id::text
    """


def county_sql() -> str:
    return f"""
    with aha as (
        select
            lpad(fcounty::text, 5, '0') as county_fips,
            case when mhsmemb = 'Yes' or sysid is not null then 1 else 0 end as system_member,
            {clean_numeric_sql('adjpd')} as adjpd
        from public.aha_fy2024
        where fcounty is not null
    ),
    county_system as (
        select
            county_fips,
            count(*) as aha_hospitals,
            sum(system_member) as system_hospitals,
            avg(system_member::numeric) as system_hospital_share,
            sum(coalesce(adjpd, 0)) as total_adjpd,
            case
                when sum(coalesce(adjpd, 0)) > 0
                then sum(coalesce(adjpd, 0) * system_member) / sum(coalesce(adjpd, 0))
            end as system_adjpd_share
        from aha
        group by county_fips
    )
    select
        c.county_fips,
        substring(c.county_fips from 1 for 2) as state_fips,
        c.census_division,
        {clean_numeric_sql('c.population')} as population,
        {clean_numeric_sql('c.pl1_ypll_rate_2019')} as ypll_2019,
        c.iv3_health_behaviors_score::numeric as health_behaviors,
        c.iv4_social_economic_factors_score::numeric as social_economic,
        c.iv2_physical_environment_score::numeric as physical_environment,
        c.iv1_medicaid_expansion_active::numeric as medicaid_expansion,
        c.dv21_premature_death_ypll_rate::numeric as dv21_ypll,
        c.ct6_hospital_deaths_age_adj_2023::numeric as ct6_hospital_deaths,
        c.mo14_ai_automate_routine_tasks_pct::numeric as mo14_pct,
        c.mo21_robotics_in_hospital_pct::numeric as mo21_pct,
        coalesce(cs.aha_hospitals, 0) as aha_hospitals,
        coalesce(cs.system_hospitals, 0) as system_hospitals,
        coalesce(cs.system_hospital_share, 0) as system_hospital_share,
        coalesce(cs.system_adjpd_share, 0) as system_adjpd_share,
        case when cs.county_fips is null or cs.system_adjpd_share is null then 1 else 0 end as system_share_missing
    from public.vw_county_file_export_wide c
    left join county_system cs
      on cs.county_fips = c.county_fips
    """


def smd_binary(p1: float, p0: float) -> float:
    pooled = math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return (p1 - p0) / pooled if pooled else float("nan")


def smd_cont(x1: pd.Series, x0: pd.Series) -> float:
    sd1 = x1.std(ddof=1)
    sd0 = x0.std(ddof=1)
    pooled = math.sqrt((sd1**2 + sd0**2) / 2)
    return (x1.mean() - x0.mean()) / pooled if pooled else float("nan")


def summarize_hospital_balance(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = [
        ("mo11_staff_scheduling", "MO11 staff-scheduling AI"),
        ("mo14_routine_tasks", "MO14 routine-task AI"),
        ("mo21_robotics", "MO21 in-hospital robotics"),
    ]
    for col, label in labels:
        sub = df.dropna(subset=[col, "system_member"]).copy()
        sys = sub[sub["system_member"] == 1][col].astype(float)
        ind = sub[sub["system_member"] == 0][col].astype(float)
        p_sys = float(sys.mean())
        p_ind = float(ind.mean())
        rows.append(
            {
                "analysis_level": "Hospital",
                "measure": label,
                "n": len(sub),
                "system_member_n": int(len(sys)),
                "non_system_n": int(len(ind)),
                "system_member_mean_pct": 100 * p_sys,
                "non_system_mean_pct": 100 * p_ind,
                "difference_pct_points": 100 * (p_sys - p_ind),
                "smd": smd_binary(p_sys, p_ind),
            }
        )
    return pd.DataFrame(rows)


def summarize_county_balance(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["mo14_any"] = (work["mo14_pct"] > 0).astype(int)
    covars = [
        ("system_adjpd_share", "System-member share of AHA adjusted patient days"),
        ("system_hospital_share", "System-member share of AHA hospitals"),
        ("aha_hospitals", "AHA hospitals in county"),
    ]
    rows = []
    for col, label in covars:
        sub = work.dropna(subset=["mo14_any", col, "dv21_ypll"]).copy()
        exposed = sub[sub["mo14_any"] == 1][col].astype(float)
        unexposed = sub[sub["mo14_any"] == 0][col].astype(float)
        rows.append(
            {
                "analysis_level": "County",
                "measure": label,
                "n": len(sub),
                "system_member_n": int(len(exposed)),
                "non_system_n": int(len(unexposed)),
                "system_member_mean_pct": 100 * exposed.mean() if "share" in col else exposed.mean(),
                "non_system_mean_pct": 100 * unexposed.mean() if "share" in col else unexposed.mean(),
                "difference_pct_points": (100 * (exposed.mean() - unexposed.mean())) if "share" in col else exposed.mean() - unexposed.mean(),
                "smd": smd_cont(exposed, unexposed),
            }
        )
    return pd.DataFrame(rows)


def design_matrix(
    df: pd.DataFrame,
    covariates: Iterable[str],
    categorical: Iterable[str] = ("census_division",),
) -> tuple[np.ndarray, list[str]]:
    blocks = []
    names = []
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
        names.append(col)
    for col in categorical:
        dummies = pd.get_dummies(df[col].fillna("Missing"), prefix=col, drop_first=True, dtype=float)
        if not dummies.empty:
            blocks.append(dummies.to_numpy(dtype=float))
            names.extend(list(dummies.columns))
    X = np.column_stack(blocks) if blocks else np.empty((len(df), 0))
    return X, names


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


def linear_aipw(df: pd.DataFrame, outcome: str, include_system_share: bool) -> dict[str, float | str]:
    base_covars = [
        "ypll_2019",
        "health_behaviors",
        "social_economic",
        "physical_environment",
        "medicaid_expansion",
        "log_population",
    ]
    covars = base_covars + (["system_adjpd_share", "system_share_missing"] if include_system_share else [])
    work = df.copy()
    work["mo14_any"] = (work["mo14_pct"] > 0).astype(float)
    work["log_population"] = np.log(pd.to_numeric(work["population"], errors="coerce").astype(float))
    needed = [outcome, "mo14_any", "state_fips"] + covars
    work = work.dropna(subset=needed).copy()
    a = work["mo14_any"].to_numpy(dtype=float)
    y = pd.to_numeric(work[outcome], errors="coerce").to_numpy(dtype=float)
    X, _ = design_matrix(work, covars)
    e = logistic_fit_predict(X, a)
    mu1, mu0 = ols_predict_potential(X, a, y)
    scores = mu1 - mu0 + (a / e) * (y - mu1) - ((1 - a) / (1 - e)) * (y - mu0)
    effect = float(np.nanmean(scores))
    se = clustered_mean_se(scores, work["state_fips"])
    return {
        "outcome": outcome,
        "specification": "Base covariates + system-member capacity share" if include_system_share else "Base covariates only",
        "n": int(len(work)),
        "treated_n": int(a.sum()),
        "control_n": int(len(work) - a.sum()),
        "estimate": effect,
        "se": se,
        "ci_lower": effect - 1.96 * se,
        "ci_upper": effect + 1.96 * se,
        "propensity_min": float(np.nanmin(e)),
        "propensity_max": float(np.nanmax(e)),
        "propensity_mean": float(np.nanmean(e)),
    }


def write_latex(balance: pd.DataFrame, aipw: pd.DataFrame) -> None:
    aipw_display = aipw[aipw["outcome"] == "dv21_ypll"].copy()
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Post-hoc AHA system-membership diagnostics and MO14 county sensitivity}",
        r"\label{tab:system_membership_sensitivity}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Panel & Contrast & N & System/Exposed mean & Non-system/Unexposed mean & SMD \\",
        r"\midrule",
    ]
    for _, row in balance.iterrows():
        panel = "A" if row["analysis_level"] == "Hospital" else "B"
        mean1 = f"{row['system_member_mean_pct']:.1f}"
        mean0 = f"{row['non_system_mean_pct']:.1f}"
        suffix = r"\%" if ("share" in row["measure"].lower() or "AI" in row["measure"] or "robotics" in row["measure"]) else ""
        lines.append(
            f"{panel} & {row['measure']} & {int(row['n']):,} & {mean1}{suffix} & {mean0}{suffix} & {row['smd']:.2f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"Panel & MO14 county sensitivity & N & Estimate & 95\% CI & -- \\",
        ]
    )
    for _, row in aipw_display.iterrows():
        outcome_label = "DV21 YPLL" if row["outcome"] == "dv21_ypll" else "CT6 hospital deaths"
        lines.append(
            f"C & {outcome_label}: {row['specification']} & {int(row['n']):,} & {row['estimate']:.1f} & "
            f"[{row['ci_lower']:.1f}, {row['ci_upper']:.1f}] & -- \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize Notes: Panel A compares hospital-level AHA system members with non-system hospitals among hospitals with non-missing workflow-AI fields. Panel B compares MO14-exposed and unexposed counties. Panel C reports a post-hoc linear-nuisance AIPW sensitivity for MO14 (any county access to AI for automating routine tasks) using the main county covariates, with and without the county share of AHA adjusted patient days in system-member hospitals. Estimates are adjusted mean differences in native outcome units and are intended as sensitivity diagnostics rather than replacements for the pre-specified primary cross-fitted Random Forest AIPW estimator.",
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )
    (ROOT / "system_membership_sensitivity_table.tex").write_text("\n".join(lines) + "\n")


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [cols] + [[("" if pd.isna(v) else f"{v:.3f}" if isinstance(v, float) else str(v)) for v in row] for row in df.to_numpy()]
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(cols))]
    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(cols))) + " |"
    out = [fmt(rows[0]), "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"]
    out.extend(fmt(row) for row in rows[1:])
    return "\n".join(out)


def main() -> None:
    with connect() as conn:
        hospitals = read_sql(conn, hospital_sql())
        counties = read_sql(conn, county_sql())

    balance = pd.concat(
        [summarize_hospital_balance(hospitals), summarize_county_balance(counties)],
        ignore_index=True,
    )
    aipw_rows = [
        linear_aipw(counties, "dv21_ypll", include_system_share=False),
        linear_aipw(counties, "dv21_ypll", include_system_share=True),
        linear_aipw(counties, "ct6_hospital_deaths", include_system_share=False),
        linear_aipw(counties, "ct6_hospital_deaths", include_system_share=True),
    ]
    aipw = pd.DataFrame(aipw_rows)

    balance.to_csv(ROOT / "system_membership_balance.csv", index=False)
    aipw.to_csv(ROOT / "system_membership_aipw_sensitivity.csv", index=False)
    write_latex(balance, aipw)

    summary = [
        "# System-Membership Sensitivity",
        "",
        "## Balance",
        markdown_table(balance),
        "",
        "## Linear-Nuisance AIPW Sensitivity",
        markdown_table(aipw),
        "",
    ]
    (ROOT / "system_membership_sensitivity_summary.md").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
