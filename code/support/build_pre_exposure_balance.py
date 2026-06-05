#!/usr/bin/env python3
"""Build reviewer-facing pre-exposure balance/pretrend tables.

This script combines:
  * 2023 AHA workflow AI indicators used in the manuscript,
  * 2022 AHA workflow AI indicators for incident-adopter contrasts, and
  * CMS 2022 public-reporting hospital quality snapshots for SEP-1 and
    pneumonia mortality.

Outputs are written to this folder as CSV, LaTeX, and Markdown notes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from sqlalchemy import create_engine


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Contrast:
    exposure: str
    exposure_label: str
    measure: str
    measure_label: str
    outcome_col: str
    snapshot_label: str
    reporting_window: str
    broad_col: str
    prior_col: str
    current_col: str
    is_pretrend: bool = False


CONTRASTS = [
    Contrast(
        exposure="wfaiss",
        exposure_label="Staff-scheduling AI",
        measure="SEP_1",
        measure_label="SEP-1 compliance (higher is better)",
        outcome_col="sep1_2022_oct",
        snapshot_label="CMS Oct. 2022 public snapshot",
        reporting_window="Jan. 1-Dec. 31, 2021",
        broad_col="wfaiss_2023",
        prior_col="wfaiss_2022",
        current_col="wfaiss_2023",
    ),
    Contrast(
        exposure="wfaiart",
        exposure_label="Routine-task AI",
        measure="MORT_30_PN",
        measure_label="Pneumonia 30-day mortality (lower is better)",
        outcome_col="pn_2022_apr",
        snapshot_label="CMS Apr. 2022 public snapshot",
        reporting_window="Jul. 1, 2017-Dec. 1, 2019",
        broad_col="wfaiart_2023",
        prior_col="wfaiart_2022",
        current_col="wfaiart_2023",
    ),
    Contrast(
        exposure="wfaiss",
        exposure_label="Staff-scheduling AI",
        measure="SEP_1_CHANGE",
        measure_label="SEP-1 change from 2019 baseline to Oct. 2022 snapshot",
        outcome_col="sep1_change_2019_to_2022_oct",
        snapshot_label="2019 baseline to CMS Oct. 2022 public snapshot",
        reporting_window="2019 baseline to Jan. 1-Dec. 31, 2021 reporting window",
        broad_col="wfaiss_2023",
        prior_col="wfaiss_2022",
        current_col="wfaiss_2023",
        is_pretrend=True,
    ),
]


def getenv(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in {None, ""} else default


def connect():
    host = getenv("POSTGRES_HOST", getenv("PGHOST", "localhost"))
    port = getenv("POSTGRES_PORT", getenv("PGPORT", "5432"))
    db = getenv("POSTGRES_DB", getenv("PGDATABASE", "Research_TEST"))
    user = getenv("POSTGRES_USER", getenv("PGUSER", "postgres"))
    password = getenv("POSTGRESQL_KEY", getenv("PGPASSWORD"))
    if not password:
        raise SystemExit("Set POSTGRESQL_KEY or PGPASSWORD before running this script.")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")


def sql_dataset() -> str:
    numeric_expr = lambda col: (
        f"case when nullif(regexp_replace({col}::text, '[^0-9.\\-]', '', 'g'), '') "
        f"is not null then nullif(regexp_replace({col}::text, '[^0-9.\\-]', '', 'g'), '')::numeric end"
    )
    return f"""
    with q as (
        select
            facility_id,
            max(score_numeric) filter (where snapshot = '2022_10' and measure_id = 'SEP_1') as sep1_2022_oct,
            max(score_numeric) filter (where snapshot = '2022_07' and measure_id = 'SEP_1') as sep1_2022_jul,
            max(score_numeric) filter (where snapshot = '2022_04' and measure_id = 'SEP_1') as sep1_2022_apr,
            max(score_numeric) filter (where snapshot = '2022_04' and measure_id = 'MORT_30_PN') as pn_2022_apr,
            max(score_numeric) filter (where snapshot = '2022_01' and measure_id = 'MORT_30_PN') as pn_2022_jan
        from public.cms_2022_hospital_quality_long
        group by facility_id
    )
    select
        v.aha_id,
        v.cms_facility_id,
        v.facility_name,
        v.state,
        v.hospital_type,
        v.hospital_ownership,
        {numeric_expr('v.adjpd')} as adjpd_numeric,
        case when v.wfaiss = '1' then 1 when v.wfaiss = '0' then 0 end as wfaiss_2023,
        case when v.wfaiart = '1' then 1 when v.wfaiart = '0' then 0 end as wfaiart_2023,
        case when v.wfaioacw = '1' then 1 when v.wfaioacw = '0' then 0 end as wfaioacw_2023,
        case when v.robohos = '1' then 1 when v.robohos = '0' then 0 end as robohos_2023,
        case when a22.wfaiss = 'Yes' then 1 when a22.wfaiss = 'No' then 0 end as wfaiss_2022,
        case when a22.wfaiart = 'Yes' then 1 when a22.wfaiart = 'No' then 0 end as wfaiart_2022,
        case when a22.wfaioacw = 'Yes' then 1 when a22.wfaioacw = 'No' then 0 end as wfaioacw_2022,
        case when a22.robohos = 'Yes' then 1 when a22.robohos = 'No' then 0 end as robohos_2022,
        {numeric_expr('v.sep_1_2019')} as sep1_2019,
        {numeric_expr('v.mort_30_pn_2019')} as pn_2019,
        q.sep1_2022_oct,
        q.sep1_2022_jul,
        q.sep1_2022_apr,
        q.pn_2022_apr,
        q.pn_2022_jan
    from public.vw_hospital_aipw_with_placebo v
    left join public.aha_fy2022 a22
        on a22.id = v.aha_id
    left join q
        on q.facility_id = v.cms_facility_id
    """


def welch_stats(x_t: pd.Series, x_c: pd.Series) -> dict[str, float]:
    nt, nc = len(x_t), len(x_c)
    mean_t, mean_c = x_t.mean(), x_c.mean()
    sd_t, sd_c = x_t.std(ddof=1), x_c.std(ddof=1)
    diff = mean_t - mean_c
    pooled = math.sqrt(((nt - 1) * sd_t**2 + (nc - 1) * sd_c**2) / (nt + nc - 2))
    smd = diff / pooled if pooled else np.nan
    se = math.sqrt(sd_t**2 / nt + sd_c**2 / nc)
    if se:
        df_num = (sd_t**2 / nt + sd_c**2 / nc) ** 2
        df_den = ((sd_t**2 / nt) ** 2 / (nt - 1)) + ((sd_c**2 / nc) ** 2 / (nc - 1))
        df = df_num / df_den if df_den else np.nan
        t_stat = diff / se
        p_value = 2 * st.t.sf(abs(t_stat), df) if not np.isnan(df) else np.nan
        crit = st.t.ppf(0.975, df) if not np.isnan(df) else np.nan
        ci_low, ci_high = diff - crit * se, diff + crit * se
    else:
        p_value, ci_low, ci_high = np.nan, np.nan, np.nan
    return {
        "n_treated": nt,
        "n_control": nc,
        "treated_mean": mean_t,
        "control_mean": mean_c,
        "treated_sd": sd_t,
        "control_sd": sd_c,
        "difference": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "smd": smd,
    }


def adjusted_model(sub: pd.DataFrame, outcome_col: str) -> dict[str, float]:
    model_df = sub[["treatment", outcome_col, "state", "hospital_type", "adjpd_numeric"]].copy()
    model_df["state"] = model_df["state"].fillna("Missing")
    model_df["hospital_type"] = model_df["hospital_type"].fillna("Missing")
    model_df["log_adjpd"] = np.log1p(model_df["adjpd_numeric"].clip(lower=0))
    model_df = model_df.dropna(subset=["treatment", outcome_col, "log_adjpd"])
    if model_df["treatment"].nunique() < 2 or len(model_df) < 50:
        return {"adjusted_difference": np.nan, "adjusted_p_value": np.nan, "adjusted_n": len(model_df)}
    try:
        fit = smf.ols(f"Q('{outcome_col}') ~ treatment + log_adjpd + C(state) + C(hospital_type)", data=model_df).fit(
            cov_type="HC1"
        )
        return {
            "adjusted_difference": fit.params.get("treatment", np.nan),
            "adjusted_p_value": fit.pvalues.get("treatment", np.nan),
            "adjusted_n": int(fit.nobs),
        }
    except Exception:
        return {"adjusted_difference": np.nan, "adjusted_p_value": np.nan, "adjusted_n": len(model_df)}


def summarize_contrast(df: pd.DataFrame, contrast: Contrast, comparison: str) -> dict[str, object]:
    if comparison == "2023 adopter vs 2023 nonadopter":
        sub = df[[contrast.outcome_col, contrast.broad_col, "state", "hospital_type", "adjpd_numeric"]].dropna(
            subset=[contrast.outcome_col, contrast.broad_col]
        )
        sub = sub.rename(columns={contrast.broad_col: "treatment"})
    elif comparison == "Incident 2023 adopter vs stable nonadopter":
        sub = df[
            (df[contrast.prior_col] == 0)
            & (df[contrast.current_col].isin([0, 1]))
            & df[contrast.outcome_col].notna()
        ][[contrast.outcome_col, contrast.current_col, "state", "hospital_type", "adjpd_numeric"]].copy()
        sub = sub.rename(columns={contrast.current_col: "treatment"})
    else:
        raise ValueError(comparison)

    sub["treatment"] = sub["treatment"].astype(int)
    x_t = sub.loc[sub["treatment"] == 1, contrast.outcome_col].astype(float)
    x_c = sub.loc[sub["treatment"] == 0, contrast.outcome_col].astype(float)
    stats = welch_stats(x_t, x_c)
    stats.update(adjusted_model(sub, contrast.outcome_col))
    stats.update(
        {
            "comparison": comparison,
            "exposure": contrast.exposure,
            "exposure_label": contrast.exposure_label,
            "measure": contrast.measure,
            "measure_label": contrast.measure_label,
            "snapshot": contrast.snapshot_label,
            "reporting_window": contrast.reporting_window,
            "interpretation": "pretrend" if contrast.is_pretrend else "pre-exposure balance",
        }
    )
    return stats


def format_num(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{float(value):,.{digits}f}"


def format_p(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_latex_table(results: pd.DataFrame) -> None:
    rows = []
    for _, row in results.iterrows():
        diff_ci = f"{format_num(row['difference'])} [{format_num(row['ci_low'])}, {format_num(row['ci_high'])}]"
        measure_window = f"{row['measure_label']}; {row['reporting_window']}"
        rows.append(
            " & ".join(
                [
                    latex_escape(row["comparison"]),
                    latex_escape(row["exposure_label"]),
                    latex_escape(measure_window),
                    format_num(row["n_treated"], 0),
                    format_num(row["n_control"], 0),
                    format_num(row["treated_mean"]),
                    format_num(row["control_mean"]),
                    diff_ci,
                    format_num(row["smd"]),
                    format_p(row["p_value"]),
                ]
            )
            + r" \\"
        )

    table = r"""\begin{table}[htbp]
\centering
\caption{Pre-exposure CMS hospital-quality balance and pretrend checks by subsequent workflow-AI adoption}
\label{tab:pre_exposure_cms_quality_balance}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.2cm}>{\raggedright\arraybackslash}p{1.75cm}>{\raggedright\arraybackslash}X>{\raggedleft\arraybackslash}p{0.75cm}>{\raggedleft\arraybackslash}p{0.75cm}>{\raggedleft\arraybackslash}p{1.0cm}>{\raggedleft\arraybackslash}p{1.0cm}>{\raggedleft\arraybackslash}p{2.2cm}>{\raggedleft\arraybackslash}p{0.8cm}>{\raggedleft\arraybackslash}p{0.8cm}@{}}
\toprule
\textbf{Comparison} & \textbf{Exposure} & \textbf{Measure/window} & $\boldsymbol{N_1}$ & $\boldsymbol{N_0}$ & \textbf{Mean$_1$} & \textbf{Mean$_0$} & \textbf{Diff. [95\% CI]} & \textbf{SMD} & $\boldsymbol{p}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}[flushleft]
\footnotesize
\item Notes: $N_1$ denotes hospitals reporting adoption in the comparison group and $N_0$ denotes comparison hospitals. The broad comparison restricts to hospitals with observed 2023 AHA item responses. The incident-adopter comparison restricts to hospitals reporting nonadoption in the 2022 AHA wave and compares those newly reporting adoption in 2023 with stable nonadopters. SEP-1 is scored so that higher values indicate better bundle adherence. Pneumonia mortality is scored so that lower values indicate better performance. Differences are adopter minus nonadopter; Welch confidence intervals and $p$-values are unadjusted balance/pretrend diagnostics, not causal estimates. The pneumonia row uses the latest 2022 CMS public snapshot with populated MORT\_30\_PN scores; later 2022 CMS hospital rows for this measure were present but not numerically scored.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    (ROOT / "pre_exposure_cms_quality_balance_table.tex").write_text(table, encoding="utf-8")


def write_notes(results: pd.DataFrame, transitions: pd.DataFrame) -> None:
    sep_inc = results[
        (results["exposure"] == "wfaiss")
        & (results["measure"] == "SEP_1_CHANGE")
        & (results["comparison"] == "Incident 2023 adopter vs stable nonadopter")
    ].iloc[0]
    pn_inc = results[
        (results["exposure"] == "wfaiart")
        & (results["comparison"] == "Incident 2023 adopter vs stable nonadopter")
    ].iloc[0]
    transition_md = dataframe_to_markdown(transitions)
    text = f"""# Pre-exposure CMS Quality Balance / Pretrend Analysis

## Design

This reviewer-responsive check compares pre-exposure CMS hospital-quality
performance between hospitals that subsequently reported workflow-AI adoption
in the 2023 AHA analytic year and those that did not.

Two contrasts are reported:

1. Observed 2023 adopter vs observed 2023 nonadopter.
2. Incident 2023 adopter vs stable nonadopter among hospitals reporting
   nonadoption in the 2022 AHA wave.

The incident-adopter contrast is the cleaner reviewer-facing comparison because
it uses the 2022 AHA wave to exclude hospitals that already reported adoption
before the manuscript exposure year.

## Key Interpretation

For the main SEP-1 mechanism check, hospitals that newly reported
staff-scheduling AI in 2023 did not show a statistically significant advantage
in the pre-exposure SEP-1 trajectory. The incident-adopter change comparison
estimated a {sep_inc['difference']:.2f}-point adopter-minus-stable-nonadopter
difference in SEP-1 change from the 2019 baseline to the October 2022 CMS
public snapshot (95% CI {sep_inc['ci_low']:.2f} to {sep_inc['ci_high']:.2f};
p={sep_inc['p_value']:.3f}; SMD={sep_inc['smd']:.2f}).

For pneumonia mortality, the latest 2022 CMS public snapshots with populated
MORT_30_PN hospital scores were January/April 2022 and used an older reporting
window. The incident routine-task AI comparison showed a
{pn_inc['difference']:.2f}-percentage-point pre-exposure difference
(95% CI {pn_inc['ci_low']:.2f} to {pn_inc['ci_high']:.2f};
p={pn_inc['p_value']:.3f}; SMD={pn_inc['smd']:.2f}). Because later 2022
MORT_30_PN rows were present but unscored, this should be described as a
baseline balance check rather than a true 2022 pneumonia pretrend. This is a
warning flag rather than a reassuring result: hospitals that later reported
routine-task AI already had modestly lower pneumonia mortality in the populated
pre-exposure public-reporting window.

## Suggested Manuscript / Response Language

We added a reviewer-responsive pre-exposure balance and trend check using CMS
hospital public-reporting snapshots released in 2022 linked to AHA workflow-AI
adoption in the 2023 analytic year. For SEP-1, the closest populated
pre-exposure snapshot was the October 2022 CMS file, which reports the
January-December 2021 performance window. Among hospitals that reported no
staff-scheduling AI in the 2022 AHA wave, hospitals newly reporting adoption in
2023 did not have a statistically significant differential SEP-1 trajectory
relative to stable nonadopters from the 2019 baseline to the 2022 public
snapshot. Pneumonia mortality was evaluated as a baseline balance check because
the later 2022 CMS MORT_30_PN rows were present but not numerically scored.
These analyses do not establish parallel trends. They reduce concern that the
primary SEP-1 association is solely an artifact of visibly superior
pre-exposure CMS performance among future staff-scheduling AI adopters, while
also showing that the pneumonia mortality association should be interpreted more
cautiously because routine-task AI adopters had modestly better pre-exposure
pneumonia mortality.

## Transition Counts

{transition_md}
"""
    (ROOT / "pre_exposure_balance_notes.md").write_text(text, encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def main() -> None:
    engine = connect()
    df = pd.read_sql(sql_dataset(), engine)
    df["sep1_change_2019_to_2022_oct"] = df["sep1_2022_oct"] - df["sep1_2019"]
    df["pn_public_update_2019_to_2022_apr"] = df["pn_2022_apr"] - df["pn_2019"]

    dataset_cols = [
        "aha_id",
        "cms_facility_id",
        "facility_name",
        "state",
        "hospital_type",
        "hospital_ownership",
        "adjpd_numeric",
        "wfaiss_2022",
        "wfaiss_2023",
        "wfaiart_2022",
        "wfaiart_2023",
        "robohos_2022",
        "robohos_2023",
        "sep1_2019",
        "sep1_2022_oct",
        "sep1_change_2019_to_2022_oct",
        "pn_2019",
        "pn_2022_apr",
        "pn_public_update_2019_to_2022_apr",
    ]
    df[dataset_cols].to_csv(ROOT / "pre_exposure_analysis_dataset.csv", index=False)

    transitions = []
    for exposure, label in [
        ("wfaiss", "Staff-scheduling AI"),
        ("wfaiart", "Routine-task AI"),
        ("robohos", "In-hospital robotics"),
    ]:
        prior = f"{exposure}_2022"
        current = f"{exposure}_2023"
        transitions.append(
            {
                "exposure": label,
                "nonadopter_2022_to_nonadopter_2023": int(((df[prior] == 0) & (df[current] == 0)).sum()),
                "nonadopter_2022_to_adopter_2023": int(((df[prior] == 0) & (df[current] == 1)).sum()),
                "adopter_2022_to_adopter_2023": int(((df[prior] == 1) & (df[current] == 1)).sum()),
                "adopter_2022_to_nonadopter_2023": int(((df[prior] == 1) & (df[current] == 0)).sum()),
            }
        )
    transitions_df = pd.DataFrame(transitions)
    transitions_df.to_csv(ROOT / "aha_2022_to_2023_transition_counts.csv", index=False)

    rows = []
    for contrast in CONTRASTS:
        for comparison in ["2023 adopter vs 2023 nonadopter", "Incident 2023 adopter vs stable nonadopter"]:
            rows.append(summarize_contrast(df, contrast, comparison))
    results = pd.DataFrame(rows)
    ordered = [
        "comparison",
        "interpretation",
        "exposure",
        "exposure_label",
        "measure",
        "measure_label",
        "snapshot",
        "reporting_window",
        "n_treated",
        "n_control",
        "treated_mean",
        "control_mean",
        "treated_sd",
        "control_sd",
        "difference",
        "ci_low",
        "ci_high",
        "p_value",
        "smd",
        "adjusted_difference",
        "adjusted_p_value",
        "adjusted_n",
    ]
    results = results[ordered]
    results.to_csv(ROOT / "pre_exposure_balance_results.csv", index=False)
    write_latex_table(results)
    write_notes(results, transitions_df)

    print(f"Wrote {ROOT / 'pre_exposure_balance_results.csv'}")
    print(f"Wrote {ROOT / 'pre_exposure_cms_quality_balance_table.tex'}")
    print(f"Wrote {ROOT / 'pre_exposure_balance_notes.md'}")


if __name__ == "__main__":
    main()
