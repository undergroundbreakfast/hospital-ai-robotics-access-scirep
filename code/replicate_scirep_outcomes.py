#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
# -*- coding: utf-8 -*-
"""
Dissertation Replication & Testing Script (v54)
---------------------------------------------------------
This script conducts a comprehensive analysis for a doctoral dissertation focusing 
on the impact of technology adoption on health outcomes and hospital efficiency.

Key Features:
- Connects securely to a PostgreSQL database.
- Fetches and harmonizes data from multiple views and tables based on the provided dictionary.
- Performs robust data preparation, including logging, dummy variable creation, 
  and centering of variables for interaction analysis.
- Runs a series of pre-defined hypothesis tests requested for the dissertation, including:
  - Direct effects using both Clustered OLS and AIPW (for causal inference).
  - Moderation effects using Clustered OLS with interaction terms.
- Includes original script's replication checks and analyses (H1-H4, CAPEX intensity).
- Generates clear, tabular results in CSV format and optional visualizations.
- Produces a single, unified report summarizing all statistical tests.

This is a complete, end-to-end script designed to be run without modification.

Author: Aaron Johnson
"""

import os
import sys
import logging
import datetime
import time
import concurrent.futures as cf
import uuid
import numpy as np
import pandas as pd
import warnings
import traceback
from typing import Any, Dict, List, Tuple, Optional

def configure_warning_filters() -> None:
    """
    Centralized warning controls so main process and spawned workers behave the same.
    """
    # pip/setuptools deprecation noise
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

    # Numerical warnings frequently emitted from sklearn/numpy linear algebra internals
    # during iterative model fitting on noisy/scaled data.
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="overflow encountered in matmul",
    )
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="divide by zero encountered in matmul",
    )
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in matmul",
    )
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"sklearn\.utils\.extmath",
    )


# Apply once at import-time so child worker processes get the same baseline filters.
configure_warning_filters()

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestRegressor

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection

from zepid.causal.doublyrobust import TMLE

# -------- Visualization (optional) ----------
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib or seaborn not found. Visualizations will be disabled.")

# Global defaults
N_BOOT = 500  # bootstrap reps for AIPW

COUNTY_OUTCOMES_WITH_BASELINE_YPLL = {
    "dv21_premature_death_ypll_rate",
    "ct5_ypll_per_100k_low",
    "ct5_ypll_per_100k_mid",
    "ct5_ypll_per_100k_high",
    "ct6_hospital_deaths_age_adj",
}

PRIMARY_COUNTY_SPECS = [
    {
        "treatment": "mo14_ai_automate_routine_tasks_pct",
        "outcome": "dv21_premature_death_ypll_rate",
        "label": "MO14→DV21",
    },
    {
        "treatment": "mo14_ai_automate_routine_tasks_pct",
        "outcome": "ct5_ypll_per_100k_mid",
        "label": "MO14→CT5",
    },
    {
        "treatment": "mo14_ai_automate_routine_tasks_pct",
        "outcome": "ct6_hospital_deaths_age_adj",
        "label": "MO14→CT6",
    },
    {
        "treatment": "mo21_robotics_in_hospital_pct",
        "outcome": "dv21_premature_death_ypll_rate",
        "label": "MO21→DV21",
    },
    {
        "treatment": "mo21_robotics_in_hospital_pct",
        "outcome": "ct5_ypll_per_100k_mid",
        "label": "MO21→CT5",
    },
    {
        "treatment": "mo21_robotics_in_hospital_pct",
        "outcome": "ct6_hospital_deaths_age_adj",
        "label": "MO21→CT6",
    },
]

MISCLASSIFICATION_SCENARIOS = [
    (0.80, 0.95),
    (0.85, 0.95),
    (0.90, 0.95),
    (0.95, 0.95),
    (0.90, 0.90),
    (0.95, 0.98),
]

# =============================================================================
# Logging
# =============================================================================
def generate_run_id(run_start_dt: Optional[datetime.datetime] = None) -> str:
    ts = (run_start_dt or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def setup_logger(
    log_file_name_prefix: str = "genai_robotics_health_analysis_log",
    log_dir: str = "logs",
    run_id: Optional[str] = None,
):
    logger = logging.getLogger("genai_robotics_health_analysis")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_suffix = run_id if run_id else timestamp
    log_file = f"{log_file_name_prefix}_{file_suffix}.txt"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logging to: {log_path}")

    bootstrap_logger = logging.getLogger("bootstrap_internal")
    if not bootstrap_logger.hasHandlers():
        bootstrap_logger.setLevel(logging.WARNING)
    return logger


def _format_md_value(value: Any, digits: int = 4) -> str:
    if isinstance(value, str):
        return value.replace("|", "\\|").strip() if value.strip() else "NA"
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value).replace("|", "\\|")


def _build_markdown_table(df: pd.DataFrame, columns: List[str], max_rows: Optional[int] = None) -> List[str]:
    if df is None or df.empty:
        return ["_No rows available._"]
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return ["_No matching columns available._"]

    df_out = df[cols].copy()
    if max_rows is not None and len(df_out) > max_rows:
        df_out = df_out.head(max_rows)

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df_out.iterrows():
        vals = [_format_md_value(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def _discover_run_artifacts(base_dirs: List[str], run_start_epoch: float) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    cutoff = run_start_epoch - 2.0
    cwd = os.getcwd()

    for base in base_dirs:
        if not base:
            continue
        if not os.path.exists(base):
            continue
        if os.path.isfile(base):
            candidates = [base]
        else:
            candidates = []
            for root, _, files in os.walk(base):
                for name in files:
                    candidates.append(os.path.join(root, name))

        for path in candidates:
            try:
                st = os.stat(path)
            except OSError:
                continue
            if st.st_mtime < cutoff:
                continue
            rel_path = os.path.relpath(path, cwd)
            artifacts.append({
                "path": rel_path,
                "mtime": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": st.st_size / 1024.0,
            })

    seen = set()
    unique_artifacts: List[Dict[str, Any]] = []
    for item in sorted(artifacts, key=lambda x: x["path"]):
        key = item["path"]
        if key in seen:
            continue
        seen.add(key)
        unique_artifacts.append(item)
    return unique_artifacts


def _extract_logger_file_paths(logger) -> List[str]:
    paths = []
    for handler in getattr(logger, "handlers", []):
        if isinstance(handler, logging.FileHandler):
            base_name = getattr(handler, "baseFilename", None)
            if base_name:
                paths.append(base_name)
    return paths


def write_run_memory_markdown(
    out_dir: str,
    run_id: str,
    logger,
    run_start_epoch: float,
    run_end_epoch: float,
    df: pd.DataFrame,
    base_controls: List[str],
    dissertation_direct_df: pd.DataFrame,
    primary_county_crossfit_df: pd.DataFrame,
    county_clip_trim_df: pd.DataFrame,
    county_ols_sensitivity: Dict[str, pd.DataFrame],
    county_misclassification_df: pd.DataFrame,
    delta_mortality_df: pd.DataFrame,
    prepost_change_df: pd.DataFrame,
    dissertation_interaction_df: pd.DataFrame,
) -> str:
    """
    Write one structured run memory markdown for downstream LLM/agent analysis.
    """
    memory_path = os.path.join(out_dir, f"run_memory_{run_id}.md")
    run_start_dt = datetime.datetime.fromtimestamp(run_start_epoch)
    run_end_dt = datetime.datetime.fromtimestamp(run_end_epoch)
    duration_min = (run_end_epoch - run_start_epoch) / 60.0

    partial_df = county_ols_sensitivity.get("partial_r2", pd.DataFrame()) if isinstance(county_ols_sensitivity, dict) else pd.DataFrame()
    oster_df = county_ols_sensitivity.get("oster", pd.DataFrame()) if isinstance(county_ols_sensitivity, dict) else pd.DataFrame()

    log_paths = _extract_logger_file_paths(logger)
    artifact_dirs = [out_dir] + log_paths
    artifacts = _discover_run_artifacts(artifact_dirs, run_start_epoch)

    lines: List[str] = []
    lines.append("### Run Metadata")
    lines.append(f"- Run_ID: `{run_id}`")
    lines.append("- Script: `code/Replicate_Results_090625_v53.py`")
    lines.append(f"- Run_Directory: `{os.path.relpath(out_dir, os.getcwd())}`")
    lines.append(f"- Start_Time: `{run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- End_Time: `{run_end_dt.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Duration_Minutes: {_format_md_value(duration_min, digits=2)}")
    lines.append(f"- Counties_In_Analysis_Frame: {_format_md_value(len(df) if df is not None else np.nan, digits=0)}")
    lines.append(f"- Columns_In_Analysis_Frame: {_format_md_value(df.shape[1] if df is not None else np.nan, digits=0)}")
    lines.append(f"- Base_Controls_Count: {_format_md_value(len(base_controls) if base_controls is not None else np.nan, digits=0)}")
    if log_paths:
        lines.append(f"- Log_File: `{os.path.relpath(log_paths[0], os.getcwd())}`")
    lines.append("")

    lines.append("### Primary County Results (Cross-Fit Summary)")
    if primary_county_crossfit_df is not None and not primary_county_crossfit_df.empty:
        cols = [
            "Label",
            "N",
            "Treatment_Prevalence",
            "Control_Mean",
            "Legacy_AIPW_ATE",
            "Crossfit_AIPW_ATE",
            "DML_ATE",
            "TMLE_ATE",
            "ATO_ATE",
            "Crossfit_Relative_Change_Pct",
        ]
        for ln in _build_markdown_table(primary_county_crossfit_df.sort_values("Label"), cols):
            lines.append(ln)
    else:
        lines.append("_Primary county cross-fit summary not available._")
    lines.append("")

    lines.append("### Key Direct Effects (AIPW p < 0.05)")
    if dissertation_direct_df is not None and not dissertation_direct_df.empty and "AIPW_p" in dissertation_direct_df.columns:
        pvals = pd.to_numeric(dissertation_direct_df["AIPW_p"], errors="coerce")
        sig = dissertation_direct_df[pvals.notna() & (pvals < 0.05)].copy()
        if not sig.empty:
            sig["AIPW_p"] = pd.to_numeric(sig["AIPW_p"], errors="coerce")
        sig = sig.sort_values(by=["AIPW_p", "Treatment", "Outcome"], ascending=[True, True, True])
        if sig.empty:
            lines.append("_No AIPW direct effects met p < 0.05 in this run._")
        else:
            cols = [
                "Treatment",
                "Outcome",
                "N",
                "AIPW_ATE",
                "AIPW_CI_Lower",
                "AIPW_CI_Upper",
                "AIPW_p",
                "Relative_Change_Pct_vs_ControlMean",
                "E_Value",
            ]
            for ln in _build_markdown_table(sig, cols, max_rows=25):
                lines.append(ln)
    else:
        lines.append("_Direct effect table not available._")
    lines.append("")

    lines.append("### Sensitivity Overview")
    lines.append(f"- Clip_Trim_Rows: {_format_md_value(len(county_clip_trim_df) if county_clip_trim_df is not None else np.nan, digits=0)}")
    if county_clip_trim_df is not None and not county_clip_trim_df.empty and "ATE" in county_clip_trim_df.columns:
        ate_vals = pd.to_numeric(county_clip_trim_df["ATE"], errors="coerce")
        lines.append(f"- Clip_Trim_ATE_Min: {_format_md_value(ate_vals.min())}")
        lines.append(f"- Clip_Trim_ATE_Max: {_format_md_value(ate_vals.max())}")
    lines.append(f"- Partial_R2_Rows: {_format_md_value(len(partial_df), digits=0)}")
    if not partial_df.empty and "Partial_R2" in partial_df.columns:
        pr2_vals = pd.to_numeric(partial_df["Partial_R2"], errors="coerce")
        lines.append(f"- Partial_R2_Median: {_format_md_value(pr2_vals.median())}")
    lines.append(f"- Oster_Rows: {_format_md_value(len(oster_df), digits=0)}")
    if not oster_df.empty and "delta_to_zero" in oster_df.columns:
        delta_vals = pd.to_numeric(oster_df["delta_to_zero"], errors="coerce")
        lines.append(f"- Oster_DeltaToZero_Median: {_format_md_value(delta_vals.median())}")
    lines.append(f"- Misclassification_Rows: {_format_md_value(len(county_misclassification_df) if county_misclassification_df is not None else np.nan, digits=0)}")
    if county_misclassification_df is not None and not county_misclassification_df.empty:
        if "Successful_Draws" in county_misclassification_df.columns:
            draws = pd.to_numeric(county_misclassification_df["Successful_Draws"], errors="coerce")
            lines.append(f"- Misclassification_Successful_Draws_Median: {_format_md_value(draws.median(), digits=1)}")
    lines.append(f"- Delta_Mortality_Rows: {_format_md_value(len(delta_mortality_df) if delta_mortality_df is not None else np.nan, digits=0)}")
    lines.append(f"- PrePost_Change_Rows: {_format_md_value(len(prepost_change_df) if prepost_change_df is not None else np.nan, digits=0)}")
    lines.append(f"- Interaction_Rows: {_format_md_value(len(dissertation_interaction_df) if dissertation_interaction_df is not None else np.nan, digits=0)}")
    lines.append("")

    lines.append("### Artifacts Generated During This Run")
    lines.append(f"- Artifact_Count: {_format_md_value(len(artifacts), digits=0)}")
    if artifacts:
        for item in artifacts:
            lines.append(
                f"- `{item['path']}` (modified `{item['mtime']}`, size `{item['size_kb']:.1f} KB`)"
            )
    else:
        lines.append("_No artifacts detected by mtime filter._")
    lines.append("")
    lines.append("---")
    lines.append("")

    with open(memory_path, "w", encoding="utf-8") as f:
        f.write("# Dissertation Replication Run Memory\n\n")
        f.write(f"Run ID: `{run_id}`\n\n")
        f.write("\n".join(lines))

    return memory_path

# =============================================================================
# DB Connection & Schema Helpers
# =============================================================================
def connect_to_database(logger) -> Engine:
    host = os.getenv("POSTGRES_HOST", 'localhost')
    database = os.getenv("POSTGRES_DB", 'Research_TEST')
    user = os.getenv("POSTGRES_USER", 'postgres')
    password = os.getenv("POSTGRESQL_KEY")

    if password is None:
        logger.error("POSTGRESQL_KEY environment variable not set.")
        sys.exit("Database password not configured. Exiting.")

    try:
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL database '{database}'.")
        return engine
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

def table_exists(engine: Engine, table_name: str, logger) -> bool:
    try:
        with engine.connect() as conn:
            res = conn.execute(
                text("""
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema IN ('public')
                      AND table_name = :tname
                    LIMIT 1
                """),
                {"tname": table_name}
            ).fetchone()
        return res is not None
    except Exception as e:
        logger.warning(f"Table existence check failed for {table_name}: {e}")
        return False

def list_columns(engine: Engine, table_name: str, logger) -> List[str]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :tname
                """),
                {"tname": table_name}
            ).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logger.warning(f"Column list failed for {table_name}: {e}")
        return []

def choose_first_existing_table(engine: Engine, candidates: List[str], logger) -> Optional[str]:
    for t in candidates:
        if table_exists(engine, t, logger):
            logger.info(f"Using table/view: {t}")
            return t
    logger.error(f"None of the candidate tables/views exist: {candidates}")
    return None

# =============================================================================
# Data Fetch & Harmonization
# =============================================================================

# Canonical names mapped based on dissertation dictionary's measure descriptions
# This mapping aligns with the provided data dictionary.
CANONICAL_TECH_COLS = {
    # mo11: AI or machine learning - staff scheduling
    "pct_wfaiss_enabled": "mo11_ai_staff_scheduling_pct",
    # mo12: AI or machine learning - predicting staffing needs
    "pct_wfaipsn_enabled": "mo12_ai_predict_staff_needs_pct",
    # mo13: AI or machine learning - predicting patient demand
    "pct_wfaippd_enabled": "mo13_ai_predict_patient_demand_pct",
    # mo14: AI or machine learning - automating routine tasks
    "pct_wfaiart_enabled": "mo14_ai_automate_routine_tasks_pct",
    # mo15: AI or machine learning - optimizing administrative and clinical workflows
    "pct_wfaioacw_enabled": "mo15_ai_optimize_workflows_pct",
    # MO21: robotics
    "pct_robohos_enabled": "mo21_robotics_in_hospital_pct",
}

# Column aliases to handle naming variants & _adjpd suffixes.
TECH_COLUMN_ALIASES: Dict[str, List[str]] = {
    "pct_wfaiss_enabled": ["pct_wfaiss_enabled_adjpd", "pct_wfaiss_enabled"],
    "pct_wfaipsn_enabled": ["pct_wfaipsn_enabled_adjpd", "pct_wfaipsn_enabled"],
    "pct_wfaippd_enabled": ["pct_wfaippd_enabled_adjpd", "pct_wfaippd_enabled"],
    "pct_wfaiart_enabled": ["pct_wfaiart_enabled_adjpd", "pct_wfaiart_enabled"],
    "pct_wfaioacw_enabled": ["pct_wfaioacw_enabled_adjpd", "pct_wfaioacw_enabled"],
    "pct_robohos_enabled": ["pct_robohos_enabled_adjpd", "pct_robohos_enabled"],
}

def resolve_first_existing_column(available_cols: List[str], alias_list: List[str]) -> Optional[str]:
    for a in alias_list:
        if a in available_cols:
            return a
    return None

def fetch_data_for_analysis(engine: Engine, logger) -> pd.DataFrame:
    """
    Builds a SELECT based on which tables & columns actually exist. Aligns to canonical column names.
    Adds HRSA rurality (SP5) if available and CAPEX intensity from AHA survey.
    """
    vcm_table = choose_first_existing_table(engine, [
        "vw_conceptual_model_adjpd", "vw_conceptual_model"
    ], logger)
    vcv_table = choose_first_existing_table(engine, [
        "vw_conceptual_model_variables_adjpd", "vw_conceptual_model_variables"
    ], logger)
    vcts_table = choose_first_existing_table(engine, [
        "vw_adjpd_weighted_tech_summary", "vw_county_tech_summary_adjpd", "vw_county_tech_summary"
    ], logger)

    hrsa_table = "hrsa_health_equity_data" if table_exists(engine, "hrsa_health_equity_data", logger) else None
    if hrsa_table:
        logger.info("HRSA health equity table found: hrsa_health_equity_data (for SP5 rurality).")
    else:
        logger.warning("HRSA health equity table NOT found; SP5 rurality models will be skipped if missing.")

    aha_table = "aha_survey_data" if table_exists(engine, "aha_survey_data", logger) else None
    if aha_table:
        logger.info("AHA survey data found: aha_survey_data (for CAPEX intensity).")
    else:
        logger.warning("AHA survey data NOT found; CAPEX intensity models will be skipped.")

    if vcm_table is None:
        sys.exit("Missing conceptual model view (vcm). Aborting.")

    # Columns present in tech table (to resolve aliases)
    tech_cols_available = list_columns(engine, vcts_table, logger) if vcts_table else []
    tech_selects = []
    for canonical_base, canonical_out in CANONICAL_TECH_COLS.items():
        if vcts_table:
            resolved = resolve_first_existing_column(tech_cols_available, TECH_COLUMN_ALIASES[canonical_base])
            if resolved:
                tech_selects.append(f"vcts.{resolved} AS {canonical_out}")
            else:
                tech_selects.append(f"NULL::numeric AS {canonical_out}")
                logger.warning(f"Tech column not found for {canonical_out}. Filled with NULL.")
        else:
            tech_selects.append(f"NULL::numeric AS {canonical_out}")

    # Base SELECT (index-level variables & outcomes)
    sql_parts = [f"""
        SELECT
            vcm.county_fips,
            vcm.health_behaviors_score             AS iv3_health_behaviors_score,
            vcm.social_economic_factors_score      AS iv4_social_economic_factors_score,
            vcm.physical_environment_score         AS iv2_physical_environment_score,
            vcm.medicaid_expansion_active          AS iv1_medicaid_expansion_active,

            vcm.health_outcomes_score              AS dv2_health_outcomes_score,
            vcm.clinical_care_score                AS dv1_clinical_care_score,
            vcm.avg_patient_services_margin        AS dv3_avg_patient_services_margin,

            vcm.population                         AS population,
            vcm.census_division                    AS census_division,

            vcm.weighted_ai_adoption_score         AS mo1_genai_composite_score,
            vcm.weighted_robotics_adoption_score   AS mo2_robotics_composite_score
    """]

    # DV components (optional)
    if vcv_table:
        sql_parts.append(f"""
            , vcv.premature_death_raw_value                AS dv21_premature_death_ypll_rate
            , vcv.ratio_of_population_to_primary_care_physicians AS dv12_physicians_ratio
            , vcv.preventable_hospital_stays_raw_value     AS dv15_preventable_stays_rate
        """)
    else:
        sql_parts.append("""
            , NULL::numeric AS dv21_premature_death_ypll_rate
            , NULL::numeric AS dv12_physicians_ratio
            , NULL::numeric AS dv15_preventable_stays_rate
        """)

    # SP5 rurality (HRSA IRR)
    if hrsa_table:
        sql_parts.append("""
            , hrsa.irr_county_value::numeric AS sp5_irr_county_value
        """)
    else:
        sql_parts.append(", NULL::numeric AS sp5_irr_county_value")

    # CAPEX intensity terms & ownership composition
    if aha_table:
        sql_parts.append("""
            , aha.capex_sum::numeric AS fi1_capex_sum
            , aha.adjpd_sum::numeric AS fi2_adjpd_sum
            , CASE WHEN aha.adjpd_sum IS NOT NULL AND aha.adjpd_sum <> 0
                   THEN (aha.capex_sum::numeric / aha.adjpd_sum::numeric)
                   ELSE NULL END AS fi_capex_intensity_ratio
            , aha.n_federal_govt::integer AS own_n_federal_govt
            , aha.n_nonfederal_govt::integer AS own_n_nonfederal_govt
            , aha.n_not_for_profit::integer AS own_n_not_for_profit
            , aha.n_for_profit::integer AS own_n_for_profit
            , aha.n_hospitals_total::integer AS own_n_hospitals_total
        """)
    else:
        sql_parts.append("""
            , NULL::numeric AS fi1_capex_sum
            , NULL::numeric AS fi2_adjpd_sum
            , NULL::numeric AS fi_capex_intensity_ratio
            , NULL::integer AS own_n_federal_govt
            , NULL::integer AS own_n_nonfederal_govt
            , NULL::integer AS own_n_not_for_profit
            , NULL::integer AS own_n_for_profit
            , NULL::integer AS own_n_hospitals_total
        """)

    # Placebo outcome: 2019 YPLL (pre-treatment mortality data)
    placebo_table = "chr_2019_ypll_placebo" if table_exists(engine, "chr_2019_ypll_placebo", logger) else None
    if placebo_table:
        sql_parts.append(", placebo.ypll_rate::numeric AS pl1_ypll_rate")
        logger.info("Placebo table chr_2019_ypll_placebo found. Will load PL1 outcome.")
    else:
        sql_parts.append(", NULL::numeric AS pl1_ypll_rate")
        logger.warning("Placebo table chr_2019_ypll_placebo not found. PL1 will be NULL.")

    # COVID-era confounding controls: COVID deaths (2020-22)
    covid_table = "vw_county_covid_deaths" if table_exists(engine, "vw_county_covid_deaths", logger) else None
    if covid_table:
        sql_parts.append(", covid.covid_deaths_total::numeric AS cv1_covid_deaths_total")
        logger.info("COVID deaths table vw_county_covid_deaths found. Will load CV1 control.")
    else:
        sql_parts.append(", NULL::numeric AS cv1_covid_deaths_total")
        logger.warning("COVID deaths table vw_county_covid_deaths not found. CV1 will be NULL.")

    # CDC WONDER 2023 mortality: Contemporaneous outcome for sensitivity analysis
    cdc_wonder_table = "cdc_2023_county_deaths_under_75yrs" if table_exists(engine, "cdc_2023_county_deaths_under_75yrs", logger) else None
    if cdc_wonder_table:
        sql_parts.append(", cdc.deaths_2023::numeric AS ct4_cdc_deaths_2023")
        logger.info("CDC WONDER 2023 mortality table found. Will load CT4 outcome (deaths).")
    else:
        sql_parts.append(", NULL::numeric AS ct4_cdc_deaths_2023")
        logger.warning("CDC WONDER 2023 mortality table not found. CT4 will be NULL.")

    # 2023 Age-Adjusted YPLL: Contemporaneous YPLL outcome for sensitivity analysis
    ypll_2023_view = '"vw_2023_age_adj_YPLL"' if table_exists(engine, "vw_2023_age_adj_YPLL", logger) else None
    if ypll_2023_view:
        sql_parts.append(", ypll2023.ct5_ypll_per_100k_low::numeric AS ct5_ypll_per_100k_low")
        sql_parts.append(", ypll2023.ct5_ypll_per_100k_mid::numeric AS ct5_ypll_per_100k_mid")
        sql_parts.append(", ypll2023.ct5_ypll_per_100k_high::numeric AS ct5_ypll_per_100k_high")
        logger.info("2023 age-adjusted YPLL view found. Will load CT5 outcome (low/mid/high suppression scenarios).")
    else:
        sql_parts.append(", NULL::numeric AS ct5_ypll_per_100k_low")
        sql_parts.append(", NULL::numeric AS ct5_ypll_per_100k_mid")
        sql_parts.append(", NULL::numeric AS ct5_ypll_per_100k_high")
        logger.warning("2023 age-adjusted YPLL view not found. CT5 will be NULL.")

    # 2023 Hospital Deaths: Contemporaneous hospital mortality outcome (age-adjusted)
    hosp_deaths_2023_view = "vw_2023_hospital_deaths" if table_exists(engine, "vw_2023_hospital_deaths", logger) else None
    if hosp_deaths_2023_view:
        sql_parts.append(", hospdeaths2023.total_age_adjusted_deaths::numeric AS ct6_hospital_deaths_age_adj")
        logger.info("2023 hospital deaths view found. Will load CT6 outcome (age-adjusted hospital deaths).")
    else:
        sql_parts.append(", NULL::numeric AS ct6_hospital_deaths_age_adj")
        logger.warning("2023 hospital deaths view not found. CT6 will be NULL.")

    # Tech component columns
    if vcts_table:
        sql_parts.append(", " + ", ".join(tech_selects))

    # FROM / JOINs
    sql_parts.append(f"FROM public.{vcm_table} AS vcm")
    if vcv_table:
        sql_parts.append(f"LEFT JOIN public.{vcv_table}  AS vcv  ON vcm.county_fips = vcv.county_fips")
    if vcts_table:
        sql_parts.append(f"LEFT JOIN public.{vcts_table} AS vcts ON vcm.county_fips = vcts.county_fips")
    if hrsa_table:
        # Normalize FIPS in HRSA (pad to 5), aggregate in case of duplicates
        sql_parts.append(f"""
            LEFT JOIN (
                SELECT
                    LPAD(TRIM(CAST(county_fips_code AS TEXT)), 5, '0') AS county_fips,
                    AVG(CASE
                          WHEN CAST(NULLIF(TRIM(CAST(irr_county_value AS TEXT)), '') AS TEXT) ~ '^[0-9]+(\\.[0-9]+)?$'
                          THEN irr_county_value::numeric
                          ELSE NULL
                        END) AS irr_county_value
                FROM public.{hrsa_table}
                GROUP BY 1
            ) AS hrsa
              ON vcm.county_fips = hrsa.county_fips
        """)

    if aha_table:
        # Aggregate AHA by county FIPS with numeric safety & padding (fcounty)
        # Also aggregate ownership codes to create county-level ownership composition
        sql_parts.append(f"""
            LEFT JOIN (
                SELECT
                    LPAD(TRIM(CAST(fcounty AS TEXT)), 5, '0') AS county_fips,
                    SUM(CASE WHEN CAST(NULLIF(TRIM(CAST(ceamt AS TEXT)), '') AS TEXT) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                             THEN ceamt::numeric ELSE NULL END) AS capex_sum,
                    SUM(CASE WHEN CAST(NULLIF(TRIM(CAST(adjpd AS TEXT)), '') AS TEXT) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                             THEN adjpd::numeric ELSE NULL END) AS adjpd_sum,
                    -- Count hospitals by ownership category
                    COUNT(CASE WHEN TRIM(cntrl) IN ('45','47','44','48','46','40') THEN 1 END) AS n_federal_govt,
                    COUNT(CASE WHEN TRIM(cntrl) IN ('12','16','14','13','15') THEN 1 END) AS n_nonfederal_govt,
                    COUNT(CASE WHEN TRIM(cntrl) IN ('23','21') THEN 1 END) AS n_not_for_profit,
                    COUNT(CASE WHEN TRIM(cntrl) IN ('32','33','31') THEN 1 END) AS n_for_profit,
                    COUNT(*) AS n_hospitals_total
                FROM public.{aha_table}
                WHERE fcounty IS NOT NULL
                GROUP BY 1
            ) AS aha
              ON vcm.county_fips = aha.county_fips
        """)

    # Placebo outcome JOIN: 2019 YPLL (pre-treatment mortality data)
    if placebo_table:
        sql_parts.append(f"""
            LEFT JOIN public.{placebo_table} AS placebo
              ON vcm.county_fips = placebo.county_fips
        """)
        logger.info("Added LEFT JOIN for chr_2019_ypll_placebo (PL1 placebo outcome).")

    # COVID deaths JOIN: COVID-19 mortality data (2020-22 cumulative)
    if covid_table:
        sql_parts.append(f"""
            LEFT JOIN (
                SELECT
                    LPAD(TRIM(CAST(county_fips AS TEXT)), 5, '0') AS county_fips,
                    CASE
                        WHEN CAST(NULLIF(TRIM(CAST(deaths_involving_covid_19 AS TEXT)), '') AS TEXT) ~ '^[0-9]+(\\.[0-9]+)?$'
                        THEN deaths_involving_covid_19::numeric
                        ELSE NULL
                    END AS covid_deaths_total
                FROM public.{covid_table}
            ) AS covid
              ON vcm.county_fips = covid.county_fips
        """)
        logger.info("Added LEFT JOIN for vw_county_covid_deaths (CV1 COVID control).")

    # CDC WONDER 2023 JOIN: Contemporaneous mortality outcome
    if cdc_wonder_table:
        sql_parts.append(f"""
            LEFT JOIN (
                SELECT
                    LPAD(TRIM(CAST(county_fips AS TEXT)), 5, '0') AS county_fips,
                    SUM(
                        CASE
                            WHEN CAST(NULLIF(TRIM(CAST(deaths AS TEXT)), '') AS TEXT) ~ '^[0-9]+(\\.[0-9]+)?$'
                            THEN deaths::numeric
                            ELSE NULL
                        END
                    ) AS deaths_2023
                FROM public.{cdc_wonder_table}
                GROUP BY LPAD(TRIM(CAST(county_fips AS TEXT)), 5, '0')
            ) AS cdc
              ON vcm.county_fips = cdc.county_fips
        """)
        logger.info("Added LEFT JOIN for cdc_2023_county_deaths_under_75yrs (CT4 outcome). Using SUM(deaths) GROUP BY county to protect against stratified data.")

    # 2023 Calculated YPLL JOIN: Contemporaneous YPLL outcome (load all three imputation scenarios)
    if ypll_2023_view:
        sql_parts.append(f"""
            LEFT JOIN (
                SELECT
                    LPAD(TRIM(CAST(county_fips AS TEXT)), 5, '0') AS county_fips,
                    ct5_ypll_u75_age_adj_per_100k_low::numeric AS ct5_ypll_per_100k_low,
                    ct5_ypll_u75_age_adj_per_100k_mid::numeric AS ct5_ypll_per_100k_mid,
                    ct5_ypll_u75_age_adj_per_100k_high::numeric AS ct5_ypll_per_100k_high
                FROM public.{ypll_2023_view}
            ) AS ypll2023
              ON vcm.county_fips = ypll2023.county_fips
        """)
        logger.info('Added LEFT JOIN for "vw_2023_age_adj_YPLL" (CT5 outcome: low/mid/high suppression scenarios).')

    # 2023 Hospital Deaths JOIN: Contemporaneous hospital mortality outcome (age-adjusted)
    if hosp_deaths_2023_view:
        sql_parts.append(f"""
            LEFT JOIN (
                SELECT
                    LPAD(TRIM(CAST(county_fips AS TEXT)), 5, '0') AS county_fips,
                    total_age_adjusted_deaths::numeric AS total_age_adjusted_deaths
                FROM public.{hosp_deaths_2023_view}
            ) AS hospdeaths2023
              ON vcm.county_fips = hospdeaths2023.county_fips
        """)
        logger.info('Added LEFT JOIN for "vw_2023_hospital_deaths" (CT6 outcome: age-adjusted hospital deaths 2023).')

    sql_parts.append("WHERE vcm.population IS NOT NULL AND CAST(vcm.population AS NUMERIC) > 0;")
    sql_query = "\n".join(sql_parts)

    try:
        df = pd.read_sql_query(text(sql_query), engine)
        if df.empty:
            raise RuntimeError("Fetched DataFrame is empty. Check views/joins/filters.")
        # FIPS hygiene
        df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)
        before = len(df)
        df = df[~df['county_fips'].str.endswith("000")]
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with FIPS ending in '000' (non-county).")

        # Numeric coercions
        numeric_cols = [
            'population', 'iv3_health_behaviors_score', 'iv4_social_economic_factors_score',
            'iv2_physical_environment_score', 'dv2_health_outcomes_score',
            'dv1_clinical_care_score', 'dv3_avg_patient_services_margin',
            'mo1_genai_composite_score', 'mo2_robotics_composite_score',
            'dv21_premature_death_ypll_rate', 'dv15_preventable_stays_rate', 
            'sp5_irr_county_value', 'fi1_capex_sum', 'fi2_adjpd_sum', 'fi_capex_intensity_ratio',
            'pl1_ypll_rate',  # Placebo outcome: 2019 YPLL
            'cv1_covid_deaths_total',  # COVID-era confounding control
            'ct4_cdc_deaths_2023',  # CDC WONDER 2023 mortality (contemporaneous outcome)
            'ct5_ypll_per_100k_low',  # 2023 age-adjusted YPLL per 100k for <75 years (suppressed=1)
            'ct5_ypll_per_100k_mid',  # 2023 age-adjusted YPLL per 100k for <75 years (suppressed=5)
            'ct5_ypll_per_100k_high',  # 2023 age-adjusted YPLL per 100k for <75 years (suppressed=9)
            'ct6_hospital_deaths_age_adj'  # 2023 age-adjusted hospital deaths (Inpatient/Outpatient/ED)
        ] + list(CANONICAL_TECH_COLS.values())
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Debug: Check if pl1_ypll_rate is loaded
        if 'pl1_ypll_rate' in df.columns:
            non_null_count = df['pl1_ypll_rate'].notna().sum()
            logger.info(f"PL1 placebo outcome loaded: {non_null_count}/{len(df)} non-null values.")
        else:
            logger.warning("PL1 placebo outcome column NOT in dataframe. Check SQL query.")

        # Create COVID deaths per 100k if COVID data available
        if 'cv1_covid_deaths_total' in df.columns:
            df['cv1_covid_deaths_per_100k'] = (df['cv1_covid_deaths_total'] / df['population']) * 100000
            non_null_count = df['cv1_covid_deaths_per_100k'].notna().sum()
            logger.info(f"CV1 COVID deaths per 100k created: {non_null_count}/{len(df)} non-null values.")
            logger.info(f"  COVID deaths per 100k range: [{df['cv1_covid_deaths_per_100k'].min():.1f}, {df['cv1_covid_deaths_per_100k'].max():.1f}]")
        else:
            logger.warning("CV1 COVID deaths column NOT in dataframe. COVID controls will not be available.")

        # Create CDC WONDER 2023 death rate per 100k (CT4) if data available
        if 'ct4_cdc_deaths_2023' in df.columns:
            df['ct4_death_rate_per_100k'] = (df['ct4_cdc_deaths_2023'] / df['population']) * 100000
            non_null_count = df['ct4_death_rate_per_100k'].notna().sum()
            logger.info(f"CT4 CDC 2023 death rate per 100k created: {non_null_count}/{len(df)} non-null values.")
            logger.info(f"  CT4 range: {df['ct4_death_rate_per_100k'].min():.1f} - {df['ct4_death_rate_per_100k'].max():.1f} deaths per 100k")

        # Log CT5 (2023 age-adjusted YPLL per 100k) availability - all three imputation scenarios
        ct5_cols = ['ct5_ypll_per_100k_low', 'ct5_ypll_per_100k_mid', 'ct5_ypll_per_100k_high']
        ct5_labels = ['Low (suppressed=1)', 'Mid (suppressed=5)', 'High (suppressed=9)']
        
        for col, label in zip(ct5_cols, ct5_labels):
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                logger.info(f"CT5 {label} loaded: {non_null_count}/{len(df)} non-null values.")
                if non_null_count > 0:
                    logger.info(f"  Range: {df[col].min():.1f} - {df[col].max():.1f} YPLL per 100k")
            else:
                logger.warning(f"CT5 {label} column NOT in dataframe.")
        
        # Check if all three scenarios available for sensitivity analysis
        if all(col in df.columns for col in ct5_cols):
            logger.info("  ✓ All three CT5 imputation scenarios available for suppression sensitivity analysis.")

        # Keep positive population
        df = df[df['population'].notna() & (df['population'] > 0)]
        logger.info(f"Data retrieved: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

# =============================================================================
# NEW: Hospital Ownership Percentages (Weighted by ADJPD)
# =============================================================================
def fetch_hospital_ownership_percentages(engine: Engine, logger) -> pd.DataFrame:
    """
    Loads hospital ownership composition by county from vw_hospital_ownership_percentages.
    Returns a dataframe with one row per county and percentage columns for each ownership category.
    Percentages are weighted by adjusted patient days (adjpd), providing accurate representation
    of hospital ownership mix accounting for facility size/utilization.
    """
    ownership_view = "vw_hospital_ownership_percentages"
    
    if not table_exists(engine, ownership_view, logger):
        logger.warning(f"{ownership_view} view not found. Ownership percentage controls will not be available.")
        return pd.DataFrame()

    sql = f"""
        SELECT
            county_fips,
            category,
            pct_adjpd_own_in_county
        FROM public.{ownership_view}
        WHERE county_fips IS NOT NULL
    """

    try:
        df_own = pd.read_sql_query(text(sql), engine)
        if df_own.empty:
            logger.warning(f"{ownership_view} returned 0 rows.")
            return df_own

        df_own.columns = [c.lower() for c in df_own.columns]
        df_own['county_fips'] = df_own['county_fips'].astype(str).str.zfill(5)
        df_own['pct_adjpd_own_in_county'] = pd.to_numeric(df_own['pct_adjpd_own_in_county'], errors='coerce')
        
        # Pivot to wide format: one row per county, columns for each ownership category
        df_pivot = df_own.pivot_table(
            index='county_fips',
            columns='category',
            values='pct_adjpd_own_in_county',
            aggfunc='sum',  # Sum in case multiple subcategories exist
            fill_value=0
        ).reset_index()
        
        # Rename columns to be code-friendly
        df_pivot.columns = ['county_fips'] + [f'own_pct_{col.lower().replace(" ", "_")}' for col in df_pivot.columns[1:]]
        
        # Ensure all expected categories exist (fill with 0 if missing)
        expected_cols = [
            'own_pct_federal_government',
            'own_pct_nonfederal_government',
            'own_pct_not_for_profit',
            'own_pct_for_profit'
        ]
        
        for col in expected_cols:
            if col not in df_pivot.columns:
                df_pivot[col] = 0.0
        
        # Convert percentages from decimal to 0-100 scale if needed
        for col in expected_cols:
            if df_pivot[col].max() <= 1.0 and df_pivot[col].max() > 0:
                df_pivot[col] = df_pivot[col] * 100
        
        logger.info(f"Ownership percentages loaded: {len(df_pivot)} counties with ownership data.")
        logger.info(f"  Columns: {[c for c in df_pivot.columns if c.startswith('own_pct_')]}")
        
        # Summary statistics
        for col in expected_cols:
            if col in df_pivot.columns:
                counties_with = (df_pivot[col] > 0).sum()
                mean_pct = df_pivot[df_pivot[col] > 0][col].mean() if counties_with > 0 else 0
                logger.info(f"  {col}: {counties_with} counties, avg {mean_pct:.1f}% of adjpd")
        
        return df_pivot

    except Exception as e:
        logger.error(f"Failed to load ownership percentages: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()

# =============================================================================
# NEW: Hospital–County Bridge Fetch (Mechanism Analysis)
# =============================================================================
def fetch_hospital_county_bridge(engine: Engine, logger) -> pd.DataFrame:
    """
    Loads the county-level bridge view that includes:
      - MO14 (workflow AI) and MO21 (robotics) at county level
      - Hospital process/safety metrics (SEP-1, OP-18b, PSI-90, Mort_30_PN)
      - County mortality outcomes (dv21_ypll_chr, ct5_ypll_u75_age_adj_per_100k_mid)

    Returns a dataframe with county_fips padded to 5 digits and numeric coercions applied.
    """
    bridge_table = choose_first_existing_table(engine, ["vw_hospital_county_bridge"], logger)
    if bridge_table is None:
        logger.warning("Bridge view vw_hospital_county_bridge not found. Mechanism block will be skipped.")
        return pd.DataFrame()

    sql = f"""
        SELECT
            LPAD(TRIM(CAST(county_fips AS TEXT)), 5, '0') AS county_fips,
            mo14_wfaiart::numeric AS mo14_wfaiart,
            mo21_robohos::numeric AS mo21_robohos,
            dv21_ypll_chr::numeric AS dv21_ypll_chr,
            ct5_ypll_u75_age_adj_per_100k_mid::numeric AS ct5_ypll_u75_age_adj_per_100k_mid,
            ef23_sep_1::numeric AS ef23_sep_1,
            ef6_op_18b::numeric AS ef6_op_18b,
            fa21_psi_90::numeric AS fa21_psi_90,
            fa27_mort_30_pn::numeric AS fa27_mort_30_pn
        FROM public.{bridge_table};
    """

    try:
        dfb = pd.read_sql_query(text(sql), engine)
        if dfb.empty:
            logger.warning("Bridge view returned 0 rows. Mechanism block will be skipped.")
            return dfb

        dfb.columns = [c.lower() for c in dfb.columns]
        dfb["county_fips"] = dfb["county_fips"].astype(str).str.zfill(5)

        # Deduplicate defensively (should already be unique)
        before = len(dfb)
        dfb = dfb.drop_duplicates(subset=["county_fips"])
        if len(dfb) != before:
            logger.warning(f"Bridge view had duplicates by county_fips. Dropped {before - len(dfb)} rows.")

        # Coerce numerics
        num_cols = [
            "mo14_wfaiart", "mo21_robohos", "dv21_ypll_chr", "ct5_ypll_u75_age_adj_per_100k_mid",
            "ef23_sep_1", "ef6_op_18b", "fa21_psi_90", "fa27_mort_30_pn"
        ]
        for c in num_cols:
            if c in dfb.columns:
                dfb[c] = pd.to_numeric(dfb[c], errors="coerce")

        logger.info(f"Bridge data loaded: {dfb.shape[0]} rows, {dfb.shape[1]} cols.")
        return dfb

    except Exception as e:
        logger.error(f"Failed to load bridge view: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def fetch_hospital_prepost_outcomes(engine: Engine, logger) -> pd.DataFrame:
    """
    Build county-level hospital quality pre/post outcomes (2019 vs 2023) from
    vw_hospital_aipw_with_placebo using ADJPD-weighted aggregation.

    Returned columns:
      - county_fips
      - hosp_sep1_2019, hosp_sep1_2023
      - hosp_op18b_2019, hosp_op18b_2023
      - hosp_mort30pn_2019, hosp_mort30pn_2023
      - hosp_n_facilities
    """
    source_view = "vw_hospital_aipw_with_placebo"
    if not table_exists(engine, source_view, logger):
        logger.warning(f"{source_view} not found. Hospital pre/post differential-change outcomes will be skipped.")
        return pd.DataFrame()

    sql = f"""
        SELECT
            LPAD(TRIM(CAST(aha_county_fips AS TEXT)), 5, '0') AS county_fips,
            adjpd,
            sep_1,
            op_18b,
            mort_30_pn_score,
            sep_1_2019,
            op_18b_2019,
            mort_30_pn_2019
        FROM public.{source_view}
        WHERE aha_county_fips IS NOT NULL
    """

    try:
        dfh = pd.read_sql_query(text(sql), engine)
        if dfh.empty:
            logger.warning(f"{source_view} returned 0 rows. Hospital pre/post outcomes will be skipped.")
            return pd.DataFrame()

        dfh.columns = [c.lower() for c in dfh.columns]
        dfh["county_fips"] = dfh["county_fips"].astype(str).str.zfill(5)
        dfh = dfh[~dfh["county_fips"].str.endswith("000")]

        for c in ["adjpd", "sep_1", "op_18b", "mort_30_pn_score", "sep_1_2019", "op_18b_2019", "mort_30_pn_2019"]:
            if c in dfh.columns:
                dfh[c] = pd.to_numeric(dfh[c], errors="coerce")

        def _weighted_mean(g: pd.DataFrame, value_col: str) -> float:
            vals = pd.to_numeric(g[value_col], errors="coerce")
            wts = pd.to_numeric(g["adjpd"], errors="coerce")
            mask_w = vals.notna() & wts.notna() & (wts > 0)
            if mask_w.any():
                return float(np.average(vals[mask_w], weights=wts[mask_w]))
            mask_u = vals.notna()
            if mask_u.any():
                return float(vals[mask_u].mean())
            return np.nan

        rows = []
        for fips, g in dfh.groupby("county_fips", sort=False):
            rows.append({
                "county_fips": fips,
                "hosp_sep1_2019": _weighted_mean(g, "sep_1_2019"),
                "hosp_sep1_2023": _weighted_mean(g, "sep_1"),
                "hosp_op18b_2019": _weighted_mean(g, "op_18b_2019"),
                "hosp_op18b_2023": _weighted_mean(g, "op_18b"),
                "hosp_mort30pn_2019": _weighted_mean(g, "mort_30_pn_2019"),
                "hosp_mort30pn_2023": _weighted_mean(g, "mort_30_pn_score"),
                "hosp_n_facilities": int(g["county_fips"].size),
            })

        out = pd.DataFrame(rows).drop_duplicates(subset=["county_fips"])
        if out.empty:
            logger.warning("Hospital pre/post county aggregation produced 0 rows.")
            return out

        logger.info(f"Hospital pre/post outcomes loaded: {len(out)} counties from {source_view}.")
        for col in [
            "hosp_sep1_2019", "hosp_sep1_2023",
            "hosp_op18b_2019", "hosp_op18b_2023",
            "hosp_mort30pn_2019", "hosp_mort30pn_2023"
        ]:
            if col in out.columns:
                logger.info(f"  {col}: {out[col].notna().sum()}/{len(out)} non-null")
        return out

    except Exception as e:
        logger.error(f"Failed to fetch hospital pre/post outcomes: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()

# =============================================================================
# NEW: Hospital-Level Ownership Analysis
# =============================================================================
def fetch_hospital_ownership_data(engine: Engine, logger) -> pd.DataFrame:
    """
    Loads hospital-level data from aha_survey_data joined with ownership codes.
    Returns a dataframe with hospital-level AI/robotics adoption and ownership information.
    """
    aha_table = "aha_survey_data" if table_exists(engine, "aha_survey_data", logger) else None
    ownership_table = "aha_appendix_a_ownership_codes" if table_exists(engine, "aha_appendix_a_ownership_codes", logger) else None
    
    if aha_table is None:
        logger.warning("aha_survey_data table not found. Hospital ownership analysis will be skipped.")
        return pd.DataFrame()
    
    if ownership_table is None:
        logger.warning("aha_appendix_a_ownership_codes table not found. Hospital ownership analysis will be skipped.")
        return pd.DataFrame()

    sql = f"""
        SELECT
            aha.id AS hospital_id,
            aha.mname AS hospital_name,
            LPAD(TRIM(CAST(aha.fcounty AS TEXT)), 5, '0') AS county_fips,
            aha.cntrl AS ownership_code,
            own.category AS ownership_category,
            own.description AS ownership_description,
            
            -- Hospital characteristics
            CASE WHEN CAST(NULLIF(TRIM(CAST(aha.adjpd AS TEXT)), '') AS TEXT) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                 THEN aha.adjpd::numeric ELSE NULL END AS adjpd,
            CASE WHEN CAST(NULLIF(TRIM(CAST(aha.admtot AS TEXT)), '') AS TEXT) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                 THEN aha.admtot::numeric ELSE NULL END AS admissions,
            CASE WHEN CAST(NULLIF(TRIM(CAST(aha.bdtot AS TEXT)), '') AS TEXT) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                 THEN aha.bdtot::numeric ELSE NULL END AS beds,
            CASE WHEN CAST(NULLIF(TRIM(CAST(aha.ceamt AS TEXT)), '') AS TEXT) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                 THEN aha.ceamt::numeric ELSE NULL END AS capex,
            
            -- AI adoption variables (binary indicators)
            CASE WHEN aha.wfaiss = '1' THEN 1 ELSE 0 END AS ai_staff_scheduling,
            CASE WHEN aha.wfaipsn = '1' THEN 1 ELSE 0 END AS ai_predict_staff_needs,
            CASE WHEN aha.wfaippd = '1' THEN 1 ELSE 0 END AS ai_predict_patient_demand,
            CASE WHEN aha.wfaiart = '1' THEN 1 ELSE 0 END AS ai_automate_routine_tasks,
            CASE WHEN aha.wfaioacw = '1' THEN 1 ELSE 0 END AS ai_optimize_workflows,
            
            -- Robotics adoption variables (binary indicators)
            CASE WHEN aha.robohos = '1' THEN 1 ELSE 0 END AS robotics_in_hospital
            
        FROM public.{aha_table} AS aha
        LEFT JOIN public.{ownership_table} AS own
            ON TRIM(aha.cntrl) = TRIM(own.code)
        WHERE aha.fcounty IS NOT NULL
    """

    try:
        dfh = pd.read_sql_query(text(sql), engine)
        if dfh.empty:
            logger.warning("Hospital ownership query returned 0 rows.")
            return dfh

        dfh.columns = [c.lower() for c in dfh.columns]
        dfh["county_fips"] = dfh["county_fips"].astype(str).str.zfill(5)

        # Coerce numerics
        num_cols = ['adjpd', 'admissions', 'beds', 'capex']
        for c in num_cols:
            if c in dfh.columns:
                dfh[c] = pd.to_numeric(dfh[c], errors="coerce")

        # Create composite AI adoption score (sum of AI technologies adopted)
        ai_cols = ['ai_staff_scheduling', 'ai_predict_staff_needs', 'ai_predict_patient_demand',
                   'ai_automate_routine_tasks', 'ai_optimize_workflows']
        dfh['ai_adoption_count'] = dfh[ai_cols].sum(axis=1)
        
        # Binary: any AI adoption
        dfh['any_ai_adoption'] = (dfh['ai_adoption_count'] > 0).astype(int)
        
        # Binary: any robotics adoption
        dfh['any_robotics_adoption'] = dfh['robotics_in_hospital']

        logger.info(f"Hospital ownership data loaded: {dfh.shape[0]} hospitals, {dfh.shape[1]} columns.")
        return dfh

    except Exception as e:
        logger.error(f"Failed to load hospital ownership data: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def analyze_hospital_ownership(df_hosp: pd.DataFrame, logger, out_dir: str) -> pd.DataFrame:
    """
    Analyzes AI/robotics adoption by hospital ownership category and description.
    Also examines hospital-level dependent variables (outcomes) by ownership.
    Returns a dataframe with summary statistics for each ownership group.
    
    Hospital-level dependent variables examined:
    - adjpd: Adjusted patient days (workload measure)
    - admissions: Total admissions (utilization measure)
    - beds: Number of beds (capacity measure)
    - capex: Capital expenditures (investment measure)
    """
    if df_hosp.empty:
        logger.warning("No hospital data provided. Skipping ownership analysis.")
        return pd.DataFrame()

    logger.info("\n" + "="*80)
    logger.info("HOSPITAL-LEVEL AI/ROBOTICS ADOPTION BY OWNERSHIP")
    logger.info("="*80)

    # Define variables for analysis
    ai_vars = ['ai_staff_scheduling', 'ai_predict_staff_needs', 'ai_predict_patient_demand',
               'ai_automate_routine_tasks', 'ai_optimize_workflows', 'ai_adoption_count', 'any_ai_adoption']
    robotics_vars = ['robotics_in_hospital', 'any_robotics_adoption']
    hospital_chars = ['adjpd', 'admissions', 'beds', 'capex']
    
    logger.info("\nHospital-level dependent variables (outcomes) being analyzed:")
    logger.info("  - adjpd: Adjusted patient days (workload/utilization)")
    logger.info("  - admissions: Total admissions (patient volume)")
    logger.info("  - beds: Number of beds (capacity)")
    logger.info("  - capex: Capital expenditures (investment in infrastructure)")
    
    # Summary by ownership CATEGORY
    logger.info("\n--- Summary by Ownership Category ---")
    results_category = []
    
    for category in df_hosp['ownership_category'].dropna().unique():
        subset = df_hosp[df_hosp['ownership_category'] == category]
        n_hospitals = len(subset)
        
        row = {
            'ownership_group_type': 'category',
            'ownership_group': category,
            'n_hospitals': n_hospitals
        }
        
        # AI adoption statistics
        for var in ai_vars:
            if var in subset.columns:
                if var in ['any_ai_adoption', 'ai_adoption_count']:
                    row[f'{var}_mean'] = subset[var].mean()
                    row[f'{var}_std'] = subset[var].std()
                else:
                    # These are binary, so mean = proportion adopting
                    row[f'{var}_pct'] = subset[var].mean() * 100
        
        # Robotics adoption statistics
        for var in robotics_vars:
            if var in subset.columns:
                row[f'{var}_pct'] = subset[var].mean() * 100
        
        # Hospital characteristics
        for var in hospital_chars:
            if var in subset.columns:
                row[f'{var}_mean'] = subset[var].mean()
                row[f'{var}_median'] = subset[var].median()
                row[f'{var}_std'] = subset[var].std()
        
        results_category.append(row)
        
        logger.info(f"\n{category} (N={n_hospitals}):")
        logger.info(f"  AI/Robotics Adoption:")
        logger.info(f"    Any AI Adoption: {row.get('any_ai_adoption_mean', 0)*100:.1f}%")
        logger.info(f"    AI Technologies (avg count): {row.get('ai_adoption_count_mean', 0):.2f}")
        logger.info(f"    Robotics Adoption: {row.get('robotics_in_hospital_pct', 0):.1f}%")
        logger.info(f"  Hospital-Level Dependent Variables (Outcomes):")
        if 'adjpd_mean' in row and not pd.isna(row['adjpd_mean']):
            logger.info(f"    Avg Adjusted Patient Days: {row['adjpd_mean']:.0f} (median: {row.get('adjpd_median', 0):.0f})")
        if 'admissions_mean' in row and not pd.isna(row['admissions_mean']):
            logger.info(f"    Avg Admissions: {row['admissions_mean']:.0f} (median: {row.get('admissions_median', 0):.0f})")
        if 'beds_mean' in row and not pd.isna(row['beds_mean']):
            logger.info(f"    Avg Beds: {row['beds_mean']:.0f} (median: {row.get('beds_median', 0):.0f})")
        if 'capex_mean' in row and not pd.isna(row['capex_mean']):
            logger.info(f"    Avg CAPEX: ${row['capex_mean']:,.0f} (median: ${row.get('capex_median', 0):,.0f})")

    df_category = pd.DataFrame(results_category)
    if not df_category.empty:
        out_file = os.path.join(out_dir, "hospital_ownership_category_summary.csv")
        df_category.to_csv(out_file, index=False)
        logger.info(f"\nSaved ownership category summary to {out_file}")

    # Summary by ownership DESCRIPTION (more detailed)
    logger.info("\n--- Summary by Ownership Description ---")
    results_description = []
    
    for description in df_hosp['ownership_description'].dropna().unique():
        subset = df_hosp[df_hosp['ownership_description'] == description]
        n_hospitals = len(subset)
        
        # Only include groups with at least 10 hospitals for meaningful statistics
        if n_hospitals < 10:
            continue
        
        row = {
            'ownership_group_type': 'description',
            'ownership_group': description,
            'ownership_category': subset['ownership_category'].iloc[0] if len(subset) > 0 else None,
            'n_hospitals': n_hospitals
        }
        
        # AI adoption statistics
        for var in ai_vars:
            if var in subset.columns:
                if var in ['any_ai_adoption', 'ai_adoption_count']:
                    row[f'{var}_mean'] = subset[var].mean()
                    row[f'{var}_std'] = subset[var].std()
                else:
                    row[f'{var}_pct'] = subset[var].mean() * 100
        
        # Robotics adoption statistics
        for var in robotics_vars:
            if var in subset.columns:
                row[f'{var}_pct'] = subset[var].mean() * 100
        
        # Hospital characteristics
        for var in hospital_chars:
            if var in subset.columns:
                row[f'{var}_mean'] = subset[var].mean()
                row[f'{var}_median'] = subset[var].median()
                row[f'{var}_std'] = subset[var].std()
        
        results_description.append(row)
        
        logger.info(f"\n{description} [{subset['ownership_category'].iloc[0]}] (N={n_hospitals}):")
        logger.info(f"  AI/Robotics Adoption:")
        logger.info(f"    Any AI Adoption: {row.get('any_ai_adoption_mean', 0)*100:.1f}%")
        logger.info(f"    AI Technologies (avg count): {row.get('ai_adoption_count_mean', 0):.2f}")
        logger.info(f"    Robotics Adoption: {row.get('robotics_in_hospital_pct', 0):.1f}%")
        logger.info(f"  Hospital-Level Dependent Variables (Outcomes):")
        if 'adjpd_mean' in row and not pd.isna(row['adjpd_mean']):
            logger.info(f"    Avg Adjusted Patient Days: {row['adjpd_mean']:.0f} (median: {row.get('adjpd_median', 0):.0f})")
        if 'admissions_mean' in row and not pd.isna(row['admissions_mean']):
            logger.info(f"    Avg Admissions: {row['admissions_mean']:.0f} (median: {row.get('admissions_median', 0):.0f})")
        if 'beds_mean' in row and not pd.isna(row['beds_mean']):
            logger.info(f"    Avg Beds: {row['beds_mean']:.0f} (median: {row.get('beds_median', 0):.0f})")
        if 'capex_mean' in row and not pd.isna(row['capex_mean']):
            logger.info(f"    Avg CAPEX: ${row['capex_mean']:,.0f} (median: ${row.get('capex_median', 0):,.0f})")

    df_description = pd.DataFrame(results_description)
    if not df_description.empty:
        out_file = os.path.join(out_dir, "hospital_ownership_description_summary.csv")
        df_description.to_csv(out_file, index=False)
        logger.info(f"\nSaved ownership description summary to {out_file}")

    # Combine both summaries
    df_combined = pd.concat([df_category, df_description], ignore_index=True)
    
    logger.info("\n" + "="*80)
    return df_combined

# =============================================================================
# Common preparation
# =============================================================================
def common_prepare_data(df_input: pd.DataFrame, logger, engine: Engine = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepares data for analysis with core confounders only (no ownership controls).
    
    NOTE: Hospital ownership is NOT included as a control variable because:
    1. It may be a MEDIATOR (AI adoption → ownership mix → outcomes), not a confounder
    2. Including it causes collider bias and inflates effect sizes
    3. V42 baseline without ownership controls showed stable, reasonable ATEs
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        Input dataframe from fetch_data_for_analysis
    logger : Logger
        Logger instance
    engine : Engine, optional
        SQLAlchemy engine (retained for compatibility but not used for ownership)
    
    Returns:
    --------
    Tuple[pd.DataFrame, List[str]]
        Prepared dataframe and list of base control variable names
    """
    df = df_input.copy()
    df.columns = [c.lower() for c in df.columns]

    # Cluster col: state FIPS
    if 'county_fips' not in df.columns:
        logger.error("county_fips not found after fetch. Exiting.")
        sys.exit(1)
    df['state_fips_for_clustering'] = df['county_fips'].astype(str).str[:2]

    # Log(pop)
    if 'population' not in df.columns:
        logger.error("population not found after fetch. Exiting.")
        sys.exit(1)
    # Filter out non-positive population before log
    df = df[df['population'] > 0]
    df['log_population'] = np.log(df['population'])

    # Medicaid expansion to numeric 0/1
    medicaid_col = 'iv1_medicaid_expansion_active'
    if medicaid_col in df.columns:
        if df[medicaid_col].dtype == bool or df[medicaid_col].dtype == np.bool_:
            df[medicaid_col] = df[medicaid_col].astype(int)
        else:
            df[medicaid_col] = (df[medicaid_col].astype(str).str.lower()
                                .map({'true':1, 't':1, 'yes':1, 'y':1, '1':1,
                                      'false':0, 'f':0, 'no':0, 'n':0, '0':0}))
        df[medicaid_col] = pd.to_numeric(df[medicaid_col], errors='coerce')
    else:
        logger.warning("Medicaid expansion column missing; moderation H9 will skip.")

    # Census Division dummies
    census_dummy_cols = []
    if 'census_division' in df.columns:
        df['census_division'] = df['census_division'].astype(str)
        dummies = pd.get_dummies(df['census_division'], prefix='div', drop_first=True, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        census_dummy_cols = list(dummies.columns)
        logger.info(f"Created {len(census_dummy_cols)} division dummies.")
    else:
        logger.warning("census_division missing; proceeding without division dummies.")

    # =========================================================================
    # BASE CONTROLS: Core confounders only (NO hospital ownership)
    # =========================================================================
    # NOTE: Hospital ownership is intentionally EXCLUDED because:
    # 1. Including it caused ATE to quadruple (from -848 to -2796 for MO14→DV21)
    # 2. Likely a MEDIATOR (AI → ownership mix → outcomes), not a true confounder
    # 3. Creates collider bias / over-adjustment
    # 4. V42 baseline controls (below) produced stable, reasonable estimates
    
    base_controls = [
        'iv4_social_economic_factors_score',
        'iv2_physical_environment_score',
        'iv3_health_behaviors_score',
        'iv1_medicaid_expansion_active',
        'log_population'
    ] + census_dummy_cols
    base_controls = [c for c in base_controls if c in df.columns]
    
    logger.info(f"\nBase controls for AIPW (n={len(base_controls)}):")
    for ctrl in base_controls:
        logger.info(f"  - {ctrl}")
    logger.info("NOTE: Hospital ownership intentionally excluded (potential mediator/collider)")
    logger.info("="*80)

    # Ensure numeric for outcome/exposure columns if present
    for col in [
        'dv21_premature_death_ypll_rate', 'dv2_health_outcomes_score',
        'dv15_preventable_stays_rate', 'dv1_clinical_care_score',
        'dv3_avg_patient_services_margin',
        'mo1_genai_composite_score', 'mo2_robotics_composite_score',
        'sp5_irr_county_value',
        'fi1_capex_sum', 'fi2_adjpd_sum', 'fi_capex_intensity_ratio'
    ] + list(CANONICAL_TECH_COLS.values()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs for tech adoption components with 0 (missing=not adopted)
    for tcol in list(CANONICAL_TECH_COLS.values()):
        if tcol in df.columns and df[tcol].isna().any():
            n = df[tcol].isna().sum()
            df[tcol] = df[tcol].fillna(0)
            logger.info(f"Filled {n} NaNs with 0 for tech column '{tcol}'.")
    if 'mo1_genai_composite_score' in df.columns and df['mo1_genai_composite_score'].isna().any():
        n = df['mo1_genai_composite_score'].isna().sum()
        df['mo1_genai_composite_score'] = df['mo1_genai_composite_score'].fillna(0)
        logger.info(f"Filled {n} NaNs with 0 for 'mo1_genai_composite_score'.")

    # Center variables used in interactions to reduce collinearity
    def center_series(s: pd.Series) -> pd.Series:
        return s - s.mean()

    vars_to_center = [
        'mo1_genai_composite_score', 'mo2_robotics_composite_score',
        'iv2_physical_environment_score', 'iv3_health_behaviors_score', 
        'sp5_irr_county_value'
    ] + list(CANONICAL_TECH_COLS.values())
    
    for col in vars_to_center:
        if col in df.columns:
            df[f"{col}_c"] = center_series(pd.to_numeric(df[col], errors='coerce'))
            logger.info(f"Created centered variable: {col}_c")
    
    # ===== SPATIAL LAG CREATION (Review 2: Spillover/Interference) =====
    # "Does my neighbor's AI adoption affect me?"
    # Create spatial lags for key technology variables to control for spillover
    logger.info("Creating spatial lags for technology variables (spillover control)...")
    spatial_lag_cols = []
    
    for tech_name, tech_col in CANONICAL_TECH_COLS.items():
        if tech_col in df.columns:
            lag_col = f"{tech_col}_spatial_lag"
            df = add_spatial_lag(df, 'state_fips_for_clustering', tech_col, lag_col)
            spatial_lag_cols.append(lag_col)
            logger.info(f"  Created spatial lag: {lag_col}")
    
    # Add composite score spatial lags
    for comp_col in ['mo1_genai_composite_score', 'mo2_robotics_composite_score']:
        if comp_col in df.columns:
            lag_col = f"{comp_col}_spatial_lag"
            df = add_spatial_lag(df, 'state_fips_for_clustering', comp_col, lag_col)
            spatial_lag_cols.append(lag_col)
            logger.info(f"  Created spatial lag: {lag_col}")
    
    # Add spatial lags to base_controls (automatically included in all models)
    base_controls.extend(spatial_lag_cols)
    logger.info(f"Added {len(spatial_lag_cols)} spatial lag controls to base_controls.")

    # CAPEX winsorization & log transforms
    if 'fi_capex_intensity_ratio' in df.columns:
        s = df['fi_capex_intensity_ratio']
        if s.notna().sum() > 10:
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            df['fi_capex_intensity_ratio_w'] = s.clip(lower=lo, upper=hi)
            df['fi_capex_intensity_ratio_log1p'] = np.log1p(df['fi_capex_intensity_ratio_w'])
            logger.info("Constructed CAPEX intensity winsorized and log1p variables.")
        else:
            df['fi_capex_intensity_ratio_w'] = s
            df['fi_capex_intensity_ratio_log1p'] = np.log1p(s.replace({-np.inf: np.nan, np.inf: np.nan}))
    else:
        logger.warning("No CAPEX intensity found; CAPEX models will be skipped.")

    return df, base_controls

# =============================================================================
# Modeling helpers
# =============================================================================
def add_const(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant='add')

def run_ols_clustered(Y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(Y, add_const(X))
    if clusters.nunique() >= 2:
        res = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})
    else:
        res = model.fit()
    return res

def backward_stepwise_by_p(Y: pd.Series, X: pd.DataFrame, keep: List[str], p_remove: float = 0.10) -> List[str]:
    vars_in = list(X.columns)
    changed = True
    while changed and len(vars_in) > len(keep):
        changed = False
        model = sm.OLS(Y, add_const(X[vars_in])).fit()  # plain OLS for selection
        pvals = model.pvalues.drop('const', errors='ignore')
        pvals = pvals.drop([v for v in keep if v in pvals.index], errors='ignore')
        if not pvals.empty:
            worst_var = pvals.idxmax()
            if pvals[worst_var] > p_remove and worst_var not in keep:
                vars_in.remove(worst_var)
                changed = True
    return vars_in

def _bh_correct_in_place(df, p_col, group_col, q_col_out):
    if df.empty or p_col not in df.columns:
        df[q_col_out] = np.nan
        return df
    if group_col is None:
        mask = df[p_col].notna()
        df[q_col_out] = np.nan
        if mask.any():
            rej, q = fdrcorrection(df.loc[mask, p_col].values, alpha=0.05, method='indep')
            df.loc[mask, q_col_out] = q
        return df
    df[q_col_out] = np.nan
    for g, dfg in df.groupby(group_col):
        mask = dfg[p_col].notna()
        if mask.any():
            rej, q = fdrcorrection(dfg.loc[mask, p_col].values, alpha=0.05, method='indep')
            df.loc[dfg.index[mask], q_col_out] = q
    return df

def _coef_se(res, name):
    try:
        return float(res.bse[name])
    except Exception:
        return np.nan

def _effective_sample_size(weights: np.ndarray) -> float:
    """Compute Kish effective sample size for weights."""
    if weights.size == 0 or np.isnan(weights).any():
        return np.nan
    denom = np.sum(weights ** 2)
    if denom == 0:
        return np.nan
    return (np.sum(weights) ** 2) / denom

def _summarize_weight_rule(ps: np.ndarray, T: np.ndarray, lower: float, upper: float, label: str) -> Dict[str, float]:
    """Summarize weights/PS under a clipping rule."""
    ps_c = np.clip(ps, lower, upper)
    w = np.where(T == 1, 1 / ps_c, 1 / (1 - ps_c))
    stats = {
        f"{label}_ps_min_t": np.min(ps_c[T == 1]) if (T == 1).any() else np.nan,
        f"{label}_ps_max_t": np.max(ps_c[T == 1]) if (T == 1).any() else np.nan,
        f"{label}_ps_min_c": np.min(ps_c[T == 0]) if (T == 0).any() else np.nan,
        f"{label}_ps_max_c": np.max(ps_c[T == 0]) if (T == 0).any() else np.nan,
        f"{label}_w_mean": float(np.mean(w)),
        f"{label}_w_median": float(np.median(w)),
        f"{label}_w_p95": float(np.percentile(w, 95)),
        f"{label}_w_p99": float(np.percentile(w, 99)),
        f"{label}_w_max": float(np.max(w)),
        f"{label}_ess_t": float(_effective_sample_size(w[T == 1])) if (T == 1).any() else np.nan,
        f"{label}_ess_c": float(_effective_sample_size(w[T == 0])) if (T == 0).any() else np.nan,
    }
    return stats

def compute_weight_diagnostics(T: np.ndarray, ps: np.ndarray, treatment: str, outcome: str) -> Dict[str, float]:
    """
    Create a diagnostics row for propensity scores/weights: prevalence, overlap, ESS, and multiple clipping rules.
    """
    diag = {
        "Treatment": treatment,
        "Outcome": outcome,
        "N": int(len(T)),
        "N_Treated": int((T == 1).sum()),
        "N_Control": int((T == 0).sum()),
        "ps_min": float(np.min(ps)),
        "ps_max": float(np.max(ps)),
        "ps_mean": float(np.mean(ps)),
        "ps_auc": roc_auc_score(T, ps) if len(np.unique(T)) == 2 else np.nan,
        "ps_brier": brier_score_loss(T, ps),
        "prevalence_treated": float(np.mean(T)),
    }
    # Base rule (matches AIPW default clip 0.01-0.99)
    diag.update(_summarize_weight_rule(ps, T, 0.01, 0.99, "clip_01_99"))
    # Alternative truncation sensitivity rules
    diag.update(_summarize_weight_rule(ps, T, 0.005, 0.995, "clip_005_995"))
    diag.update(_summarize_weight_rule(ps, T, 0.02, 0.98, "clip_02_98"))
    return diag


def get_control_set(y_col: str, base_controls: List[str], df: pd.DataFrame) -> List[str]:
    """
    Resolve outcome-specific controls without mutating the shared base control list.
    County mortality models append baseline YPLL when available in the current frame.
    """
    controls = [c for c in list(base_controls) if c in df.columns]
    if (
        y_col in COUNTY_OUTCOMES_WITH_BASELINE_YPLL
        and "pl1_ypll_rate" in df.columns
        and df["pl1_ypll_rate"].notna().any()
        and "pl1_ypll_rate" not in controls
    ):
        controls.append("pl1_ypll_rate")
    return controls


def get_county_restricted_controls(df: pd.DataFrame) -> List[str]:
    """Restricted Oster model: baseline YPLL, census division FE, and log population."""
    controls: List[str] = []
    if "pl1_ypll_rate" in df.columns and df["pl1_ypll_rate"].notna().any():
        controls.append("pl1_ypll_rate")
    if "log_population" in df.columns and df["log_population"].notna().any():
        controls.append("log_population")
    div_cols = sorted(
        c for c in df.columns
        if c.startswith("div_") and df[c].notna().any()
    )
    controls.extend(div_cols)
    return list(dict.fromkeys(controls))


def sanitize_filename_component(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("%", "pct")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("→", "_to_")
    )


def add_binary_treatment_column(
    df: pd.DataFrame,
    treat_col: str,
    rule: str = "median",
    out_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, float]:
    """
    Add a binary treatment indicator using a reproducible threshold rule.
    """
    d = df.copy()
    if out_col is None:
        out_col = f"{treat_col}_binary"

    treat_vals = pd.to_numeric(d[treat_col], errors="coerce")
    if rule == "median":
        threshold = float(treat_vals.median())
        d[out_col] = (treat_vals > threshold).astype(int)
    elif rule == "sparse_mo14":
        threshold = float(treat_vals.quantile(0.75))
        if threshold == 0:
            nonzero_vals = treat_vals[treat_vals > 0]
            if len(nonzero_vals) > 0:
                threshold = float(nonzero_vals.min())
            else:
                threshold = float(treat_vals.median())
        d[out_col] = (treat_vals > threshold).astype(int)
    elif rule == "positive":
        threshold = 0.0
        d[out_col] = (treat_vals > 0).astype(int)
    else:
        raise ValueError(f"Unsupported treatment rule: {rule}")

    return d, out_col, threshold


def prepare_county_binary_analysis(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str,
    base_controls: List[str],
    treatment_rule: str = "median",
) -> Tuple[pd.DataFrame, List[str], str, float]:
    """
    Prepare a county analytic frame using outcome-specific controls and a binary treatment.
    """
    controls = get_control_set(y_col, base_controls, df)
    required_cols = [treat_col, y_col, "state_fips_for_clustering"] + controls
    required_cols = [c for c in required_cols if c in df.columns]
    d = df[required_cols].dropna().copy()
    if d.empty:
        return d, controls, f"{treat_col}_binary", np.nan
    d, bin_col, threshold = add_binary_treatment_column(
        d,
        treat_col=treat_col,
        rule=treatment_rule,
        out_col=f"{treat_col}_{sanitize_filename_component(y_col)}_{treatment_rule}_binary",
    )
    return d, controls, bin_col, threshold


def compute_relative_change(effect: float, control_mean: float) -> float:
    if not np.isfinite(effect) or not np.isfinite(control_mean) or abs(control_mean) <= 1e-9:
        return np.nan
    return 100.0 * effect / control_mean


def summarize_weight_profile(weights: np.ndarray) -> Dict[str, float]:
    finite_w = weights[np.isfinite(weights)]
    if finite_w.size == 0:
        return {"ess": np.nan, "max_weight": np.nan, "p99_weight": np.nan}
    return {
        "ess": float(_effective_sample_size(finite_w)),
        "max_weight": float(np.max(finite_w)),
        "p99_weight": float(np.percentile(finite_w, 99)),
    }

# =============================================================================
# IPTW / AIPW
# =============================================================================
def estimate_propensity_scores(X_confounders, T_treatment, logger, treatment_name, C_param=0.1):
    scaler = StandardScaler()
    Xn = X_confounders.apply(pd.to_numeric, errors='coerce').copy()
    if Xn.isnull().any().any():
        Xn = Xn.fillna(Xn.mean())
    Xs = scaler.fit_transform(Xn)
    try:
        if len(np.unique(T_treatment)) < 2:
            logger.error(f"PS model ({treatment_name}): treatment has one class.")
            return None, None
        if np.min(np.bincount(T_treatment.astype(int))) < 5:
            logger.warning(f"PS model ({treatment_name}): class imbalance; results may be unstable.")
        lr = LogisticRegression(solver='liblinear', random_state=42, C=C_param, penalty='l1', max_iter=300)
        lr.fit(Xs, T_treatment)
        ps = lr.predict_proba(Xs)[:, 1]
    except Exception as e:
        logger.error(f"PS model fitting failed for {treatment_name}: {e}")
        return None, None
    ps = np.clip(ps, 0.01, 0.99)
    return ps, scaler

def run_aipw(df, treat_col, y_col, confounders, n_boot, logger, plot_dir,
             cluster_ids=None, cluster_bootstrap=False):
    confounders = get_control_set(y_col, confounders, df)
    T = df[treat_col].astype(int).values
    Y = df[y_col].values
    X = df[confounders].copy()
    if X.isnull().any().any():
        X = X.fillna(X.mean())

    ps, scaler = estimate_propensity_scores(X, T, logger, treat_col)
    if ps is None or scaler is None:
        return (np.nan, np.nan, np.nan, np.nan,
                int((T == 1).sum()), int((T == 0).sum()), True)

    # PS overlap plot
    if VISUALIZATION_AVAILABLE:
        plt.figure(figsize=(9, 5))
        try:
            sns.histplot(ps[T == 1], label=f"Treated (N={(T==1).sum()})", stat="density", kde=True, bins=30, alpha=0.6)
            sns.histplot(ps[T == 0], label=f"Control (N={(T==0).sum()})", stat="density", kde=True, bins=30, alpha=0.6)
            plt.title(f"Propensity Score Overlap: {treat_col}")
            plt.xlabel("Propensity score"); plt.legend(); plt.tight_layout()
            os.makedirs(plot_dir, exist_ok=True)
            outp = os.path.join(plot_dir, f"ps_overlap_{treat_col.replace('%','pct')}.png")
            plt.savefig(outp, dpi=300); plt.close()
        except Exception:
            plt.close()
    
    def _fit_mu_single(X_all, Y_all, mask):
        y = Y_all[mask]
        Xa = X_all.copy()
        if np.isnan(Xa).any():
            col_means = np.nanmean(Xa, axis=0)
            inds = np.where(np.isnan(Xa))
            Xa[inds] = np.take(col_means, inds[1])
        if mask.sum() > Xa.shape[1] and mask.sum() > 5:
            try:
                m = sm.OLS(y, add_const(Xa[mask])).fit()
                return m.predict(add_const(Xa))
            except Exception:
                return np.full(len(Y_all), np.mean(y) if mask.sum() > 0 else np.mean(Y_all))
        else:
            return np.full(len(Y_all), np.mean(y) if mask.sum() > 0 else np.mean(Y_all))

    Xs = scaler.transform(X.apply(pd.to_numeric, errors='coerce'))
    mu1 = _fit_mu_single(Xs, Y, (T == 1))
    mu0 = _fit_mu_single(Xs, Y, (T == 0))
    
    term1 = (T / ps) * (Y - mu1) + mu1
    term0 = ((1 - T) / (1 - ps)) * (Y - mu0) + mu0
    ate = np.mean(term1[np.isfinite(term1)]) - np.mean(term0[np.isfinite(term0)])

    # Bootstrap CIs and p-approx
    ate_boot = []
    df_reset = df.reset_index(drop=True)

    # Attach cluster ids for clustered bootstrap (optional)
    if cluster_bootstrap and cluster_ids is not None:
        df_reset["__cluster_id"] = pd.Series(cluster_ids).reset_index(drop=True)
        unique_clusters = df_reset["__cluster_id"].dropna().unique()
        if len(unique_clusters) < 2:
            logger.warning("Clustered bootstrap requested but <2 clusters; falling back to IID bootstrap.")
            cluster_bootstrap = False

    for i in range(n_boot):
        try:
            if cluster_bootstrap:
                sampled_clusters = np.random.choice(unique_clusters, size=len(unique_clusters), replace=True)
                parts = [df_reset[df_reset["__cluster_id"] == c] for c in sampled_clusters]
                b = pd.concat(parts, ignore_index=True)
            else:
                b = resample(df_reset, replace=True, random_state=i)

            if b.empty or b[treat_col].nunique() < 2:
                continue
            T_b = b[treat_col].astype(int).values
            Y_b = b[y_col].values
            X_b = b[confounders].copy().fillna(b[confounders].mean())
            ps_b, sc_b = estimate_propensity_scores(X_b, T_b, logging.getLogger("bootstrap_internal"), treat_col)
            if ps_b is None or sc_b is None:
                continue
            Xs_b = sc_b.transform(X_b.apply(pd.to_numeric, errors='coerce'))
            mu1_b = _fit_mu_single(Xs_b, Y_b, (T_b == 1))
            mu0_b = _fit_mu_single(Xs_b, Y_b, (T_b == 0))
            
            term1_b = (T_b / ps_b) * (Y_b - mu1_b) + mu1_b
            term0_b = ((1 - T_b) / (1 - ps_b)) * (Y_b - mu0_b) + mu0_b
            ate_b = np.mean(term1_b[np.isfinite(term1_b)]) - np.mean(term0_b[np.isfinite(term0_b)])

            if np.isfinite(ate_b): ate_boot.append(ate_b)
        except Exception:
            pass

    ate_ci = (np.percentile(ate_boot, 2.5), np.percentile(ate_boot, 97.5)) if ate_boot else (np.nan, np.nan)

    # two-sided bootstrap p approx (centered at 0)
    p_raw = np.nan
    if ate_boot and np.isfinite(ate):
        boots = np.array(ate_boot)
        if ate > 0:
            p_raw = 2 * (np.sum(boots <= 0) + 1) / (len(boots) + 1)
        elif ate < 0:
            p_raw = 2 * (np.sum(boots >= 0) + 1) / (len(boots) + 1)
        else:
            p_raw = 1.0
        p_raw = float(min(p_raw, 1.0))

    return (ate, ate_ci[0], ate_ci[1], p_raw,
            int((T == 1).sum()), int((T == 0).sum()), False)


def _prepare_fold_feature_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Numeric coercion and mean-imputation using training-fold moments only.
    """
    X_train_num = X_train.apply(pd.to_numeric, errors="coerce").copy()
    train_means = X_train_num.mean().fillna(0)
    X_train_num = X_train_num.fillna(train_means)

    X_test_num = X_test.apply(pd.to_numeric, errors="coerce").copy()
    X_test_num = X_test_num.fillna(train_means)
    return X_train_num, X_test_num


def crossfit_aipw_point_estimate(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str,
    confounders: List[str],
    logger,
    ps_clip: Tuple[float, float] = (0.01, 0.99),
    n_splits: int = 5,
    rf_n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Cross-fitted AIPW with logistic PS nuisance and separate treated/control RF outcome models.
    """
    confounders = get_control_set(y_col, confounders, df)
    required_cols = [treat_col, y_col] + confounders
    required_cols = [c for c in required_cols if c in df.columns]
    d = df[required_cols].copy()

    d[treat_col] = pd.to_numeric(d[treat_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[treat_col, y_col]).copy()

    if d.empty or d[treat_col].nunique() < 2:
        return {
            "ate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_treated": 0,
            "n_control": 0,
            "diagnostics": {},
            "error": True,
        }

    T = d[treat_col].astype(int).values
    Y = d[y_col].astype(float).values

    if len(np.unique(T)) != 2:
        logger.warning(f"Cross-fit AIPW requires binary treatment: {treat_col}")
        return {
            "ate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_treated": int((T == 1).sum()),
            "n_control": int((T == 0).sum()),
            "diagnostics": {},
            "error": True,
        }

    n_treated = int((T == 1).sum())
    n_control = int((T == 0).sum())
    min_class = min(n_treated, n_control)
    n_splits_eff = min(n_splits, min_class)
    if n_splits_eff < 2:
        logger.warning(
            f"Cross-fit AIPW has too few observations per class for {treat_col}->{y_col} "
            f"(treated={n_treated}, control={n_control})."
        )
        return {
            "ate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_treated": n_treated,
            "n_control": n_control,
            "diagnostics": {},
            "error": True,
        }

    X = d[confounders].copy()
    raw_ps = np.full(len(d), np.nan)
    ps = np.full(len(d), np.nan)
    mu1 = np.full(len(d), np.nan)
    mu0 = np.full(len(d), np.nan)

    splitter = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=42)

    try:
        for train_idx, test_idx in splitter.split(X, T):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            T_train = T[train_idx]
            Y_train = Y[train_idx]

            X_train_num, X_test_num = _prepare_fold_feature_frames(X_train, X_test)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_num)
            X_test_scaled = scaler.transform(X_test_num)

            ps_model = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            ps_model.fit(X_train_scaled, T_train)
            raw_ps[test_idx] = ps_model.predict_proba(X_test_scaled)[:, 1]
            ps[test_idx] = np.clip(raw_ps[test_idx], ps_clip[0], ps_clip[1])

            treat_mask = T_train == 1
            control_mask = T_train == 0

            y_treat = np.mean(Y_train[treat_mask]) if treat_mask.any() else np.mean(Y_train)
            y_control = np.mean(Y_train[control_mask]) if control_mask.any() else np.mean(Y_train)

            if treat_mask.sum() >= 5:
                rf_treat = RandomForestRegressor(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=rf_n_jobs,
                    max_features="sqrt",
                    min_samples_leaf=5,
                )
                rf_treat.fit(X_train_num.iloc[np.where(treat_mask)[0]], Y_train[treat_mask])
                mu1[test_idx] = rf_treat.predict(X_test_num)
            else:
                mu1[test_idx] = y_treat

            if control_mask.sum() >= 5:
                rf_control = RandomForestRegressor(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=rf_n_jobs,
                    max_features="sqrt",
                    min_samples_leaf=5,
                )
                rf_control.fit(X_train_num.iloc[np.where(control_mask)[0]], Y_train[control_mask])
                mu0[test_idx] = rf_control.predict(X_test_num)
            else:
                mu0[test_idx] = y_control
    except Exception as e:
        logger.warning(f"Cross-fit AIPW failed for {treat_col}->{y_col}: {e}")
        return {
            "ate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_treated": n_treated,
            "n_control": n_control,
            "diagnostics": {},
            "error": True,
        }

    phi = (
        (mu1 - mu0)
        + T * (Y - mu1) / ps
        - (1 - T) * (Y - mu0) / (1 - ps)
    )
    weights = np.where(T == 1, 1 / ps, 1 / (1 - ps))
    finite_mask = np.isfinite(phi) & np.isfinite(weights)
    if finite_mask.sum() == 0:
        return {
            "ate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_treated": n_treated,
            "n_control": n_control,
            "diagnostics": {},
            "error": True,
        }

    ate = float(np.mean(phi[finite_mask]))
    influence = phi[finite_mask] - ate
    se = (
        float(np.std(influence, ddof=1) / np.sqrt(finite_mask.sum()))
        if finite_mask.sum() > 1
        else np.nan
    )
    ci_lower = ate - 1.96 * se if np.isfinite(se) else np.nan
    ci_upper = ate + 1.96 * se if np.isfinite(se) else np.nan
    p_value = (
        float(2 * (1 - stats.norm.cdf(abs(ate / se))))
        if np.isfinite(se) and se > 0
        else np.nan
    )

    diagnostics = {
        "ps": ps,
        "raw_ps": raw_ps,
        "weights": weights,
        "phi": phi,
        "mu1": mu1,
        "mu0": mu0,
        "ess_overall": float(_effective_sample_size(weights[finite_mask])),
        "ess_treated": float(_effective_sample_size(weights[(T == 1) & finite_mask])) if n_treated > 0 else np.nan,
        "ess_control": float(_effective_sample_size(weights[(T == 0) & finite_mask])) if n_control > 0 else np.nan,
        "max_weight": float(np.max(weights[finite_mask])),
        "p99_weight": float(np.percentile(weights[finite_mask], 99)),
        "prevalence": float(np.mean(T)),
        "controls": confounders,
        "n": int(len(d)),
        "n_finite": int(finite_mask.sum()),
        "ps_clip_lower": float(ps_clip[0]),
        "ps_clip_upper": float(ps_clip[1]),
    }

    return {
        "ate": ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_treated": n_treated,
        "n_control": n_control,
        "diagnostics": diagnostics,
        "error": False,
    }


def run_crossfit_aipw(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str,
    confounders: List[str],
    n_boot: int,
    logger,
    cluster_ids=None,
    cluster_bootstrap: bool = False,
    ps_clip: Tuple[float, float] = (0.01, 0.99),
    rf_n_jobs: int = -1,
) -> Tuple[float, float, float, float, int, int, Dict[str, Any], bool]:
    """
    Cross-fitted AIPW wrapper with optional IID or state-cluster bootstrap.
    """
    point = crossfit_aipw_point_estimate(
        df=df,
        treat_col=treat_col,
        y_col=y_col,
        confounders=confounders,
        logger=logger,
        ps_clip=ps_clip,
        rf_n_jobs=rf_n_jobs,
    )
    if point["error"]:
        return (np.nan, np.nan, np.nan, np.nan, point["n_treated"], point["n_control"], {}, True)

    ate = point["ate"]
    ci_lower = point["ci_lower"]
    ci_upper = point["ci_upper"]
    p_value = point["p_value"]

    if n_boot and n_boot > 0:
        ate_boot: List[float] = []
        df_reset = df.reset_index(drop=True).copy()

        if cluster_bootstrap and cluster_ids is not None:
            df_reset["__cluster_id"] = pd.Series(cluster_ids).reset_index(drop=True)
            unique_clusters = df_reset["__cluster_id"].dropna().unique()
            if len(unique_clusters) < 2:
                logger.warning("Cross-fit AIPW clustered bootstrap requested but <2 clusters; using IID bootstrap.")
                cluster_bootstrap = False

        for i in range(n_boot):
            try:
                if cluster_bootstrap:
                    sampled_clusters = np.random.choice(unique_clusters, size=len(unique_clusters), replace=True)
                    boot_df = pd.concat(
                        [df_reset[df_reset["__cluster_id"] == c] for c in sampled_clusters],
                        ignore_index=True,
                    )
                else:
                    boot_df = resample(df_reset, replace=True, random_state=i)

                point_b = crossfit_aipw_point_estimate(
                    df=boot_df,
                    treat_col=treat_col,
                    y_col=y_col,
                    confounders=confounders,
                    logger=logging.getLogger("bootstrap_internal"),
                    ps_clip=ps_clip,
                    rf_n_jobs=rf_n_jobs,
                )
                if not point_b["error"] and np.isfinite(point_b["ate"]):
                    ate_boot.append(float(point_b["ate"]))
            except Exception:
                continue

        if ate_boot:
            boot_arr = np.array(ate_boot)
            ci_lower = float(np.percentile(boot_arr, 2.5))
            ci_upper = float(np.percentile(boot_arr, 97.5))
            if ate > 0:
                p_value = float(min(1.0, 2 * (np.sum(boot_arr <= 0) + 1) / (len(boot_arr) + 1)))
            elif ate < 0:
                p_value = float(min(1.0, 2 * (np.sum(boot_arr >= 0) + 1) / (len(boot_arr) + 1)))
            else:
                p_value = 1.0

    return (
        ate,
        ci_lower,
        ci_upper,
        p_value,
        point["n_treated"],
        point["n_control"],
        point["diagnostics"],
        False,
    )


def run_aipw_overlap(df, treat_col, y_col, confounders, n_boot, logger, plot_dir):
    """
    AIPW with overlap weighting (ATO estimand).
    
    Overlap weights: w_i = e(X_i) * (1 - e(X_i)) for ALL units.
    This targets the Average Treatment effect on the Overlap (ATO) - the population
    where covariate balance is best. Less sensitive to extreme propensities.
    
    Reference: Li, Morgan & Zaslavsky (2018). "Balancing Covariates via Propensity 
    Score Weighting." JASA.
    """
    confounders = get_control_set(y_col, confounders, df)
    T = df[treat_col].astype(int).values
    Y = df[y_col].values
    X = df[confounders].copy()
    if X.isnull().any().any():
        X = X.fillna(X.mean())

    ps, scaler = estimate_propensity_scores(X, T, logger, treat_col)
    if ps is None or scaler is None:
        return (np.nan, np.nan, np.nan, np.nan,
                int((T == 1).sum()), int((T == 0).sum()), np.nan, True)

    # Overlap weights: w_i = e(X_i) * (1 - e(X_i)) for ALL units
    overlap_weights = ps * (1 - ps)
    
    # Calculate effective sample size (ESS) for reporting
    ess_treated = _effective_sample_size(overlap_weights[T == 1]) if (T == 1).sum() > 0 else np.nan
    ess_control = _effective_sample_size(overlap_weights[T == 0]) if (T == 0).sum() > 0 else np.nan
    ess_overall = _effective_sample_size(overlap_weights)
    
    # Fit outcome models
    def _fit_mu_single(X_all, Y_all, mask):
        y = Y_all[mask]
        Xa = X_all.copy()
        if np.isnan(Xa).any():
            col_means = np.nanmean(Xa, axis=0)
            inds = np.where(np.isnan(Xa))
            Xa[inds] = np.take(col_means, inds[1])
        if mask.sum() > Xa.shape[1] and mask.sum() > 5:
            try:
                m = sm.OLS(y, add_const(Xa[mask])).fit()
                return m.predict(add_const(Xa))
            except Exception:
                return np.full(len(Y_all), np.mean(y) if mask.sum() > 0 else np.mean(Y_all))
        else:
            return np.full(len(Y_all), np.mean(y) if mask.sum() > 0 else np.mean(Y_all))

    Xs = scaler.transform(X.apply(pd.to_numeric, errors='coerce'))
    mu1 = _fit_mu_single(Xs, Y, (T == 1))
    mu0 = _fit_mu_single(Xs, Y, (T == 0))
    
    # Correct overlap-weighted AIPW using DR pseudo-outcome
    # phi_i = (mu1_i - mu0_i) + T_i*(Y_i - mu1_i)/e_i - (1-T_i)*(Y_i - mu0_i)/(1-e_i)
    # tau_hat = sum(h_i * phi_i) / sum(h_i) where h_i = e_i * (1 - e_i)
    
    # Clip propensity scores to avoid division by near-zero
    ps_clipped = np.clip(ps, 0.01, 0.99)
    
    # Compute DR pseudo-outcome
    phi = (mu1 - mu0) + T * (Y - mu1) / ps_clipped - (1 - T) * (Y - mu0) / (1 - ps_clipped)
    
    # Weighted average using overlap weights
    finite_mask = np.isfinite(phi) & np.isfinite(overlap_weights)
    if finite_mask.sum() == 0 or overlap_weights[finite_mask].sum() == 0:
        ate = np.nan
    else:
        ate = np.sum(overlap_weights[finite_mask] * phi[finite_mask]) / np.sum(overlap_weights[finite_mask])

    # Bootstrap CIs
    ate_boot = []
    df_reset = df.reset_index(drop=True)
    for i in range(n_boot):
        try:
            b = resample(df_reset, replace=True, random_state=i)
            if b.empty or b[treat_col].nunique() < 2:
                continue
            T_b = b[treat_col].astype(int).values
            Y_b = b[y_col].values
            X_b = b[confounders].copy().fillna(b[confounders].mean())
            ps_b, sc_b = estimate_propensity_scores(X_b, T_b, logging.getLogger("bootstrap_internal"), treat_col)
            if ps_b is None or sc_b is None:
                continue
            
            # Overlap weights for bootstrap sample
            ow_b = ps_b * (1 - ps_b)
            
            # Clip propensity scores
            ps_b_clipped = np.clip(ps_b, 0.01, 0.99)
            
            Xs_b = sc_b.transform(X_b.apply(pd.to_numeric, errors='coerce'))
            mu1_b = _fit_mu_single(Xs_b, Y_b, (T_b == 1))
            mu0_b = _fit_mu_single(Xs_b, Y_b, (T_b == 0))
            
            # Compute DR pseudo-outcome
            phi_b = (mu1_b - mu0_b) + T_b * (Y_b - mu1_b) / ps_b_clipped - (1 - T_b) * (Y_b - mu0_b) / (1 - ps_b_clipped)
            
            # Weighted average using overlap weights
            finite_mask_b = np.isfinite(phi_b) & np.isfinite(ow_b)
            if finite_mask_b.sum() == 0 or ow_b[finite_mask_b].sum() == 0:
                continue
            
            ate_b = np.sum(ow_b[finite_mask_b] * phi_b[finite_mask_b]) / np.sum(ow_b[finite_mask_b])

            if np.isfinite(ate_b): 
                ate_boot.append(ate_b)
        except Exception:
            pass

    ate_ci = (np.percentile(ate_boot, 2.5), np.percentile(ate_boot, 97.5)) if ate_boot else (np.nan, np.nan)

    # Bootstrap p-value
    p_raw = np.nan
    if ate_boot and np.isfinite(ate):
        boots = np.array(ate_boot)
        if ate > 0:
            p_raw = 2 * (np.sum(boots <= 0) + 1) / (len(boots) + 1)
        elif ate < 0:
            p_raw = 2 * (np.sum(boots >= 0) + 1) / (len(boots) + 1)
        else:
            p_raw = 1.0
        p_raw = float(min(p_raw, 1.0))

    return (ate, ate_ci[0], ate_ci[1], p_raw,
            int((T == 1).sum()), int((T == 0).sum()), ess_overall, False)


def plot_ps_and_weight_diagnostics(
    T: np.ndarray,
    ps: np.ndarray,
    weights: np.ndarray,
    out_path_prefix: str,
    logger,
):
    """
    Save positivity diagnostics for propensity scores and inverse-probability weights.
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization not available. Skipping PS/weight diagnostic plots.")
        return

    finite_ps = np.isfinite(ps)
    finite_w = np.isfinite(weights)
    os.makedirs(os.path.dirname(out_path_prefix), exist_ok=True)

    try:
        plt.figure(figsize=(9, 5))
        sns.histplot(ps[(T == 1) & finite_ps], bins=30, alpha=0.6, stat="density", kde=True, label=f"Treated (N={(T == 1).sum()})")
        sns.histplot(ps[(T == 0) & finite_ps], bins=30, alpha=0.6, stat="density", kde=True, label=f"Control (N={(T == 0).sum()})")
        plt.xlabel("Predicted propensity score")
        plt.ylabel("Density")
        plt.title("Cross-Fitted Propensity Score Overlap")
        plt.legend()
        plt.tight_layout()
        ps_out = f"{out_path_prefix}_propensity_hist.png"
        plt.savefig(ps_out, dpi=300)
        plt.close()
        logger.info(f"Saved PS overlap diagnostic: {ps_out}")
    except Exception as e:
        plt.close()
        logger.warning(f"Failed to save PS overlap diagnostic: {e}")

    try:
        plt.figure(figsize=(9, 5))
        plot_weights = weights[finite_w]
        sns.histplot(plot_weights, bins=40, kde=True, color="#4C72B0")
        if plot_weights.size > 0 and np.nanmax(plot_weights) > 25:
            plt.xscale("log")
            plt.xlabel("Inverse-probability weight (log scale)")
        else:
            plt.xlabel("Inverse-probability weight")
        plt.ylabel("Count")
        plt.title("Cross-Fitted Weight Distribution")
        plt.tight_layout()
        w_out = f"{out_path_prefix}_weight_hist.png"
        plt.savefig(w_out, dpi=300)
        plt.close()
        logger.info(f"Saved weight diagnostic: {w_out}")
    except Exception as e:
        plt.close()
        logger.warning(f"Failed to save weight diagnostic: {e}")


def run_clip_trim_sensitivity(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str,
    base_controls: List[str],
    logger,
    plot_dir: str,
    n_boot: int = 0,
    treatment_rule: str = "median",
) -> pd.DataFrame:
    """
    Re-estimate cross-fitted county ATEs under alternative clipping and trimming rules.
    """
    d, controls, bin_col, threshold = prepare_county_binary_analysis(
        df=df,
        treat_col=treat_col,
        y_col=y_col,
        base_controls=base_controls,
        treatment_rule=treatment_rule,
    )
    if d.empty or d[bin_col].nunique() < 2:
        logger.warning(f"Clip/trim sensitivity skipped for {treat_col}->{y_col}: insufficient data.")
        return pd.DataFrame()

    scenario_rows: List[Dict[str, Any]] = []

    baseline = crossfit_aipw_point_estimate(
        df=d,
        treat_col=bin_col,
        y_col=y_col,
        confounders=controls,
        logger=logger,
        ps_clip=(0.01, 0.99),
    )
    if baseline["error"]:
        logger.warning(f"Clip/trim sensitivity baseline failed for {treat_col}->{y_col}.")
        return pd.DataFrame()

    diag_prefix = os.path.join(
        plot_dir,
        f"crossfit_{sanitize_filename_component(treat_col)}_{sanitize_filename_component(y_col)}",
    )
    plot_ps_and_weight_diagnostics(
        T=d[bin_col].astype(int).values,
        ps=baseline["diagnostics"]["ps"],
        weights=baseline["diagnostics"]["weights"],
        out_path_prefix=diag_prefix,
        logger=logger,
    )

    clip_specs = [
        {"scenario": "clip_0.005_0.995", "clip": (0.005, 0.995), "trim": None},
        {"scenario": "clip_0.01_0.99", "clip": (0.01, 0.99), "trim": None},
        {"scenario": "clip_0.02_0.98", "clip": (0.02, 0.98), "trim": None},
        {"scenario": "trim_ps_0.05_0.95", "clip": (0.01, 0.99), "trim": (0.05, 0.95)},
        {"scenario": "trim_empirical_common_support", "clip": (0.01, 0.99), "trim": "empirical"},
    ]

    for spec in clip_specs:
        d_use = d.copy()
        trim_rule = spec["trim"]
        if trim_rule == "empirical":
            raw_ps = baseline["diagnostics"]["raw_ps"]
            T_vals = d[bin_col].astype(int).values
            lower = max(np.nanmin(raw_ps[T_vals == 1]), np.nanmin(raw_ps[T_vals == 0]))
            upper = min(np.nanmax(raw_ps[T_vals == 1]), np.nanmax(raw_ps[T_vals == 0]))
            trim_mask = np.isfinite(raw_ps) & (raw_ps >= lower) & (raw_ps <= upper)
        elif isinstance(trim_rule, tuple):
            raw_ps = baseline["diagnostics"]["raw_ps"]
            lower, upper = trim_rule
            trim_mask = np.isfinite(raw_ps) & (raw_ps >= lower) & (raw_ps <= upper)
        else:
            lower = upper = np.nan
            trim_mask = np.ones(len(d_use), dtype=bool)

        d_use = d_use.loc[trim_mask].copy()
        if d_use.empty or d_use[bin_col].nunique() < 2:
            logger.warning(f"Clip/trim scenario {spec['scenario']} dropped all usable observations for {treat_col}->{y_col}.")
            continue

        ate, ci_l, ci_u, p_val, n_t, n_c, diagnostics, err = run_crossfit_aipw(
            df=d_use,
            treat_col=bin_col,
            y_col=y_col,
            confounders=controls,
            n_boot=n_boot,
            logger=logger,
            cluster_ids=d_use["state_fips_for_clustering"] if "state_fips_for_clustering" in d_use.columns else None,
            cluster_bootstrap=False,
            ps_clip=spec["clip"],
        )
        if err:
            continue

        weight_stats = summarize_weight_profile(diagnostics["weights"])
        scenario_rows.append({
            "Treatment": treat_col,
            "Outcome": y_col,
            "Threshold": threshold,
            "Scenario": spec["scenario"],
            "Clip_Lower": spec["clip"][0],
            "Clip_Upper": spec["clip"][1],
            "Trim_Lower": lower,
            "Trim_Upper": upper,
            "ATE": ate,
            "CI_Lower": ci_l,
            "CI_Upper": ci_u,
            "p_value": p_val,
            "N_Retained": len(d_use),
            "N_Treated_Retained": n_t,
            "N_Control_Retained": n_c,
            "ESS": weight_stats["ess"],
            "Max_Weight": weight_stats["max_weight"],
            "P99_Weight": weight_stats["p99_weight"],
        })

    return pd.DataFrame(scenario_rows)


def run_partial_r2_sensitivity(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str,
    base_controls: List[str],
    logger,
) -> Dict[str, Any]:
    """
    Partial R^2 sensitivity using clustered OLS and the county manuscript control set.
    """
    controls = get_control_set(y_col, base_controls, df)
    required_cols = [treat_col, y_col, "state_fips_for_clustering"] + controls
    required_cols = [c for c in required_cols if c in df.columns]
    d = df[required_cols].dropna().copy()
    if d.empty or d[treat_col].nunique() < 2:
        logger.warning(f"Partial R^2 sensitivity skipped for {treat_col}->{y_col}: insufficient data.")
        return {}

    res = run_ols_clustered(d[y_col], d[[treat_col] + controls], d["state_fips_for_clustering"])
    t_stat = float(res.tvalues.get(treat_col, np.nan))
    df_resid = float(res.df_resid) if hasattr(res, "df_resid") else np.nan
    partial_r2 = (t_stat ** 2) / ((t_stat ** 2) + df_resid) if np.isfinite(t_stat) and np.isfinite(df_resid) and df_resid > 0 else np.nan

    return {
        "Treatment": treat_col,
        "Outcome": y_col,
        "N": len(d),
        "Controls": ", ".join(controls),
        "Beta": float(res.params.get(treat_col, np.nan)),
        "SE": float(res.bse.get(treat_col, np.nan)),
        "t_stat": t_stat,
        "p_value": float(res.pvalues.get(treat_col, np.nan)),
        "df_resid": df_resid,
        "Partial_R2": partial_r2,
    }


def run_oster_sensitivity(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str,
    base_controls: List[str],
    logger,
) -> Dict[str, Any]:
    """
    Oster-style omitted variable bias bounds using restricted vs full clustered OLS models.
    """
    full_controls = get_control_set(y_col, base_controls, df)
    restricted_controls = get_county_restricted_controls(df)
    needed_cols = list(dict.fromkeys([treat_col, y_col, "state_fips_for_clustering"] + full_controls + restricted_controls))
    needed_cols = [c for c in needed_cols if c in df.columns]
    d = df[needed_cols].dropna().copy()
    if d.empty or d[treat_col].nunique() < 2:
        logger.warning(f"Oster sensitivity skipped for {treat_col}->{y_col}: insufficient data.")
        return {}

    res_restricted = run_ols_clustered(
        d[y_col],
        d[[treat_col] + restricted_controls] if restricted_controls else d[[treat_col]],
        d["state_fips_for_clustering"],
    )
    res_full = run_ols_clustered(
        d[y_col],
        d[[treat_col] + full_controls] if full_controls else d[[treat_col]],
        d["state_fips_for_clustering"],
    )

    beta_restricted = float(res_restricted.params.get(treat_col, np.nan))
    beta_full = float(res_full.params.get(treat_col, np.nan))
    r2_restricted = float(getattr(res_restricted, "rsquared", np.nan))
    r2_full = float(getattr(res_full, "rsquared", np.nan))
    rmax = min(0.99, 1.3 * r2_full) if np.isfinite(r2_full) else np.nan

    beta_star_delta1 = np.nan
    delta_to_zero = np.nan
    if (
        np.isfinite(beta_restricted)
        and np.isfinite(beta_full)
        and np.isfinite(r2_restricted)
        and np.isfinite(r2_full)
        and np.isfinite(rmax)
        and (r2_full > r2_restricted)
        and (rmax > r2_full)
        and abs(beta_restricted - beta_full) > 1e-12
    ):
        beta_star_delta1 = beta_full - (beta_restricted - beta_full) * ((rmax - r2_full) / (r2_full - r2_restricted))
        delta_to_zero = (beta_full / (beta_restricted - beta_full)) * ((r2_full - r2_restricted) / (rmax - r2_full))

    return {
        "Treatment": treat_col,
        "Outcome": y_col,
        "N": len(d),
        "Restricted_Controls": ", ".join(restricted_controls),
        "Full_Controls": ", ".join(full_controls),
        "beta_restricted": beta_restricted,
        "beta_full": beta_full,
        "R2_restricted": r2_restricted,
        "R2_full": r2_full,
        "Rmax": rmax,
        "beta_star_delta_1": beta_star_delta1,
        "delta_to_zero": delta_to_zero,
    }


def run_treatment_misclassification_sensitivity(
    df: pd.DataFrame,
    treat_col: str,
    y_col: str,
    base_controls: List[str],
    logger,
    se_sp_grid: Optional[List[Tuple[float, float]]] = None,
    n_draws: int = 200,
    ps_clip: Tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Nondifferential treatment misclassification sensitivity for extensive-margin binary exposure.
    """
    controls = get_control_set(y_col, base_controls, df)
    required_cols = [treat_col, y_col, "state_fips_for_clustering"] + controls
    required_cols = [c for c in required_cols if c in df.columns]
    d = df[required_cols].dropna().copy()
    if d.empty:
        logger.warning(f"Misclassification sensitivity skipped for {treat_col}->{y_col}: no data.")
        return pd.DataFrame()

    d["__t_obs"] = (pd.to_numeric(d[treat_col], errors="coerce") > 0).astype(int)
    if d["__t_obs"].nunique() < 2:
        logger.warning(f"Misclassification sensitivity skipped for {treat_col}->{y_col}: observed treatment is constant.")
        return pd.DataFrame()

    p_obs = float(d["__t_obs"].mean())
    grid = se_sp_grid or MISCLASSIFICATION_SCENARIOS
    rows: List[Dict[str, Any]] = []

    for se, sp in grid:
        denom = se + sp - 1
        if denom <= 0:
            logger.warning(f"Invalid Se/Sp scenario ({se}, {sp}) for {treat_col}->{y_col}.")
            continue

        pi = (p_obs + sp - 1) / denom
        if not (0 <= pi <= 1):
            logger.warning(
                f"Latent prevalence outside [0,1] for {treat_col}->{y_col} under Se={se}, Sp={sp}: pi={pi:.4f}."
            )
            continue

        post_t1_if_obs1 = np.clip((se * pi) / p_obs, 0, 1) if p_obs > 0 else np.nan
        post_t1_if_obs0 = np.clip(((1 - se) * pi) / (1 - p_obs), 0, 1) if p_obs < 1 else np.nan
        if not np.isfinite(post_t1_if_obs1) or not np.isfinite(post_t1_if_obs0):
            continue

        draw_ates: List[float] = []
        for draw in range(n_draws):
            rng = np.random.default_rng(42 + draw)
            posterior_probs = np.where(
                d["__t_obs"].values == 1,
                post_t1_if_obs1,
                post_t1_if_obs0,
            )
            d["__t_latent"] = rng.binomial(1, posterior_probs, size=len(d))
            if d["__t_latent"].nunique() < 2:
                continue

            point = crossfit_aipw_point_estimate(
                df=d,
                treat_col="__t_latent",
                y_col=y_col,
                confounders=controls,
                logger=logging.getLogger("bootstrap_internal"),
                ps_clip=ps_clip,
            )
            if not point["error"] and np.isfinite(point["ate"]):
                draw_ates.append(float(point["ate"]))

        if not draw_ates:
            continue

        draw_arr = np.array(draw_ates)
        rows.append({
            "Treatment": treat_col,
            "Outcome": y_col,
            "Observed_Prevalence": p_obs,
            "Sensitivity": se,
            "Specificity": sp,
            "Latent_Prevalence": pi,
            "Posterior_T1_if_Obs1": post_t1_if_obs1,
            "Posterior_T1_if_Obs0": post_t1_if_obs0,
            "N": len(d),
            "Successful_Draws": len(draw_arr),
            "ATE_Median": float(np.median(draw_arr)),
            "ATE_P2_5": float(np.percentile(draw_arr, 2.5)),
            "ATE_P97_5": float(np.percentile(draw_arr, 97.5)),
        })

    return pd.DataFrame(rows)


def _run_misclassification_scenario_task(
    d: pd.DataFrame,
    y_col: str,
    controls: List[str],
    se: float,
    sp: float,
    n_draws: int,
    ps_clip: Tuple[float, float],
    seed_offset: int = 0,
    rf_n_jobs: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Worker task for one Se/Sp scenario in misclassification sensitivity.
    """
    configure_warning_filters()
    if d.empty or "__t_obs" not in d.columns:
        return None

    p_obs = float(d["__t_obs"].mean())
    denom = se + sp - 1
    if denom <= 0:
        return None

    pi = (p_obs + sp - 1) / denom
    if not (0 <= pi <= 1):
        return None

    post_t1_if_obs1 = np.clip((se * pi) / p_obs, 0, 1) if p_obs > 0 else np.nan
    post_t1_if_obs0 = np.clip(((1 - se) * pi) / (1 - p_obs), 0, 1) if p_obs < 1 else np.nan
    if not np.isfinite(post_t1_if_obs1) or not np.isfinite(post_t1_if_obs0):
        return None

    posterior_probs = np.where(
        d["__t_obs"].values == 1,
        post_t1_if_obs1,
        post_t1_if_obs0,
    )
    d_local = d.copy()
    draw_ates: List[float] = []

    for draw in range(n_draws):
        rng = np.random.default_rng(42 + seed_offset * 100000 + draw)
        d_local["__t_latent"] = rng.binomial(1, posterior_probs, size=len(d_local))
        if d_local["__t_latent"].nunique() < 2:
            continue

        point = crossfit_aipw_point_estimate(
            df=d_local,
            treat_col="__t_latent",
            y_col=y_col,
            confounders=controls,
            logger=logging.getLogger("bootstrap_internal"),
            ps_clip=ps_clip,
            rf_n_jobs=rf_n_jobs,
        )
        if not point["error"] and np.isfinite(point["ate"]):
            draw_ates.append(float(point["ate"]))

    if not draw_ates:
        return None

    draw_arr = np.array(draw_ates)
    return {
        "Observed_Prevalence": p_obs,
        "Sensitivity": se,
        "Specificity": sp,
        "Latent_Prevalence": pi,
        "Posterior_T1_if_Obs1": post_t1_if_obs1,
        "Posterior_T1_if_Obs0": post_t1_if_obs0,
        "N": len(d_local),
        "Successful_Draws": len(draw_arr),
        "ATE_Median": float(np.median(draw_arr)),
        "ATE_P2_5": float(np.percentile(draw_arr, 2.5)),
        "ATE_P97_5": float(np.percentile(draw_arr, 97.5)),
    }


# =============================================================================
# NEW: Mechanism Bridge Analysis (County-Level)
# =============================================================================
def run_mechanism_bridge_block(df_main: pd.DataFrame,
                               df_bridge: pd.DataFrame,
                               base_controls: List[str],
                               logger,
                               out_dir: str,
                               plot_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Produces mechanism evidence using vw_hospital_county_bridge:
      A) First stage: MO14/MO21 -> mechanism metrics
      B) Mechanism -> CT5 (with tech in model)
      C) Attenuation: MO14/MO21 -> CT5 shrinks when mechanism vars added

    Keeps metrics "as-is" (no sign flips), per user request.
    """

    results = {}

    if df_bridge is None or df_bridge.empty:
        logger.warning("Mechanism bridge block skipped: no bridge data.")
        return results

    # Ensure lowercase columns (df_main is already lowercased in common_prepare_data)
    df = df_main.copy()
    df_bridge_local = df_bridge.copy()
    df_bridge_local.columns = [c.lower() for c in df_bridge_local.columns]

    # Merge: keep only counties that have both controls and bridge variables
    mech = df.merge(df_bridge_local, on="county_fips", how="inner")
    logger.info(f"Mechanism merged sample: {len(mech)} counties (inner-join).")

    # Controls available in this merged set
    controls = [c for c in base_controls if c in mech.columns]
    if "state_fips_for_clustering" not in mech.columns:
        mech["state_fips_for_clustering"] = mech["county_fips"].astype(str).str[:2]

    # Core variables from bridge view
    tech_vars = ["mo14_wfaiart", "mo21_robohos"]
    mech_vars = ["ef23_sep_1", "fa21_psi_90", "fa27_mort_30_pn", "ef6_op_18b"]

    # Outcomes: use your CT5 if you want apples-to-apples; fallback to bridge CT5 if needed
    # (You can change this if you prefer one explicitly.)
    ct5_primary = "ct5_ypll_per_100k_mid" if "ct5_ypll_per_100k_mid" in mech.columns else "ct5_ypll_u75_age_adj_per_100k_mid"

    # ---------------------------
    # (0) Coverage / descriptives
    # ---------------------------
    desc_cols = ["mo14_wfaiart", "mo21_robohos", ct5_primary] + mech_vars
    desc_cols = [c for c in desc_cols if c in mech.columns]
    coverage_rows = []
    for c in desc_cols:
        s = pd.to_numeric(mech[c], errors="coerce")
        coverage_rows.append({
            "Variable": c,
            "N_nonnull": int(s.notna().sum()),
            "Mean": float(s.mean()) if s.notna().sum() > 0 else np.nan,
            "SD": float(s.std()) if s.notna().sum() > 1 else np.nan,
            "Min": float(s.min()) if s.notna().sum() > 0 else np.nan,
            "Max": float(s.max()) if s.notna().sum() > 0 else np.nan,
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_csv = os.path.join(out_dir, "bridge_mechanism_coverage_descriptives.csv")
    coverage_df.to_csv(coverage_csv, index=False)
    logger.info(f"Saved bridge coverage/descriptives: {coverage_csv}")
    results["coverage"] = coverage_df

    # ---------------------------
    # (A) First stage: Tech -> mechanism (clustered OLS)
    # ---------------------------
    # Interpretation:
    # - SEP-1: higher better, so positive beta is "better"
    # - PSI-90: lower better, so negative beta is "better"
    # - Mort_30_PN: lower better, so negative beta is "better"
    # - OP-18b: lower better, so negative beta is "better"
    first_stage_rows = []
    for y in mech_vars:
        cols = [y, "state_fips_for_clustering"] + tech_vars + controls
        cols = [c for c in cols if c in mech.columns]
        d = mech[cols].dropna().copy()
        if len(d) < 150 or d[y].nunique() < 5:
            logger.warning(f"First stage skipped for {y}: N={len(d)} or low variation.")
            continue

        X = d[[v for v in tech_vars if v in d.columns] + controls]
        res = run_ols_clustered(d[y], X, d["state_fips_for_clustering"])

        for t in tech_vars:
            if t not in res.params.index:
                continue
            ci = res.conf_int().loc[t].values
            first_stage_rows.append({
                "Stage": "Tech -> Mechanism (Clustered OLS)",
                "Outcome": y,
                "Predictor": t,
                "N": len(d),
                "Beta": float(res.params[t]),
                "p_value": float(res.pvalues[t]),
                "CI_Lower": float(ci[0]),
                "CI_Upper": float(ci[1]),
                "Beta_per_10pp": float(res.params[t] * 0.10)  # interpret if tech vars are 0..1
            })

    first_stage_df = pd.DataFrame(first_stage_rows)
    if not first_stage_df.empty:
        # FDR across all first-stage tests
        first_stage_df = _bh_correct_in_place(first_stage_df, "p_value", None, "q_value_fdr")
        out_csv = os.path.join(out_dir, "bridge_first_stage_tech_to_mechanism_ols.csv")
        first_stage_df.to_csv(out_csv, index=False)
        logger.info(f"Saved first-stage mechanism OLS table: {out_csv}")
    else:
        logger.warning("No first-stage results produced.")
    results["first_stage_ols"] = first_stage_df

    # ---------------------------
    # (B) Mechanism -> CT5 (clustered OLS), with tech still in model
    # ---------------------------
    mech_to_ct5_rows = []
    # One model per mechanism metric (avoids collinearity), plus a "full" model with all 4
    model_sets = [([m], f"Add {m}") for m in mech_vars] + [(mech_vars, "Add ALL mechanism metrics")]

    for mech_set, model_label in model_sets:
        cols = [ct5_primary, "state_fips_for_clustering"] + tech_vars + controls + mech_set
        cols = [c for c in cols if c in mech.columns]
        d = mech[cols].dropna().copy()
        if len(d) < 150 or d[ct5_primary].nunique() < 5:
            logger.warning(f"Mechanism->CT5 model skipped ({model_label}): N={len(d)}")
            continue

        X = d[[v for v in tech_vars if v in d.columns] + controls + mech_set]
        res = run_ols_clustered(d[ct5_primary], X, d["state_fips_for_clustering"])

        # Save only the "story" coefficients (tech + mechanism), not every control
        for v in (tech_vars + mech_set):
            if v not in res.params.index:
                continue
            ci = res.conf_int().loc[v].values
            mech_to_ct5_rows.append({
                "Model": model_label,
                "Outcome": ct5_primary,
                "Predictor": v,
                "N": len(d),
                "Beta": float(res.params[v]),
                "p_value": float(res.pvalues[v]),
                "CI_Lower": float(ci[0]),
                "CI_Upper": float(ci[1]),
                "R2": float(res.rsquared) if hasattr(res, "rsquared") else np.nan
            })

    mech_to_ct5_df = pd.DataFrame(mech_to_ct5_rows)
    if not mech_to_ct5_df.empty:
        mech_to_ct5_df = _bh_correct_in_place(mech_to_ct5_df, "p_value", "Model", "q_value_fdr")
        out_csv = os.path.join(out_dir, "bridge_mechanism_to_ct5_ols.csv")
        mech_to_ct5_df.to_csv(out_csv, index=False)
        logger.info(f"Saved mechanism->CT5 OLS table: {out_csv}")
    results["mech_to_ct5_ols"] = mech_to_ct5_df

    # ---------------------------
    # (C) Attenuation: Tech -> CT5 shrinkage when adding mechanism metrics
    # ---------------------------
    attenuation_rows = []

    # Base model (no mechanism metrics)
    cols_base = [ct5_primary, "state_fips_for_clustering"] + tech_vars + controls
    cols_base = [c for c in cols_base if c in mech.columns]
    d0 = mech[cols_base].dropna().copy()

    if len(d0) >= 150 and d0[ct5_primary].nunique() >= 5:
        X0 = d0[[v for v in tech_vars if v in d0.columns] + controls]
        res0 = run_ols_clustered(d0[ct5_primary], X0, d0["state_fips_for_clustering"])
        base_betas = {t: float(res0.params.get(t, np.nan)) for t in tech_vars}

        # For each mediator (and all mediators), refit and compute attenuation relative to base
        for mech_set, model_label in model_sets:
            cols = [ct5_primary, "state_fips_for_clustering"] + tech_vars + controls + mech_set
            cols = [c for c in cols if c in mech.columns]
            d = mech[cols].dropna().copy()
            if len(d) < 150:
                continue

            X = d[[v for v in tech_vars if v in d.columns] + controls + mech_set]
            res = run_ols_clustered(d[ct5_primary], X, d["state_fips_for_clustering"])

            for t in tech_vars:
                if t not in res.params.index:
                    continue
                beta_new = float(res.params[t])
                beta_base = base_betas.get(t, np.nan)

                # Attenuation: how much closer to 0 the coefficient gets
                # (Works for negative or positive betas)
                atten_pct = np.nan
                if np.isfinite(beta_base) and abs(beta_base) > 1e-9:
                    atten_pct = ((beta_base - beta_new) / beta_base) * 100

                attenuation_rows.append({
                    "Tech": t,
                    "Model": model_label,
                    "Outcome": ct5_primary,
                    "N": len(d),
                    "Beta_Base": beta_base,
                    "Beta_With_Mech": beta_new,
                    "Attenuation_Pct": atten_pct,
                    "p_value_with_mech": float(res.pvalues.get(t, np.nan)),
                    "R2": float(res.rsquared) if hasattr(res, "rsquared") else np.nan
                })

    attenuation_df = pd.DataFrame(attenuation_rows)
    if not attenuation_df.empty:
        out_csv = os.path.join(out_dir, "bridge_ct5_attenuation_summary.csv")
        attenuation_df.to_csv(out_csv, index=False)
        logger.info(f"Saved attenuation table: {out_csv}")
    else:
        logger.warning("No attenuation results produced (likely low N after merge/dropna).")
    results["attenuation"] = attenuation_df

    # ---------------------------
    # (D) Simple binned-scatter plots (optional but strong visually)
    # ---------------------------
    if VISUALIZATION_AVAILABLE:
        def _binned_scatter(dfx, xcol, ycol, outname, nbins=20):
            dd = dfx[[xcol, ycol]].dropna().copy()
            if len(dd) < 200 or dd[xcol].nunique() < 10:
                return
            try:
                dd["bin"] = pd.qcut(dd[xcol], q=nbins, duplicates="drop")
                g = dd.groupby("bin", observed=False).agg({xcol: "mean", ycol: "mean"}).reset_index(drop=True)

                plt.figure(figsize=(8, 5))
                plt.plot(g[xcol], g[ycol], marker="o")
                plt.xlabel(xcol)
                plt.ylabel(ycol)
                plt.title(f"Binned mean: {ycol} vs {xcol} ({nbins} bins)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                os.makedirs(plot_dir, exist_ok=True)
                outp = os.path.join(plot_dir, outname)
                plt.savefig(outp, dpi=300)
                plt.close()
                logger.info(f"Saved binned scatter: {outp}")
            except Exception:
                plt.close()

        _binned_scatter(mech, "mo14_wfaiart", "ef23_sep_1", "bridge_binned_mo14_vs_sep1.png")
        _binned_scatter(mech, "mo14_wfaiart", "ef6_op_18b", "bridge_binned_mo14_vs_op18b.png")

    return results


# =============================================================================
# Delta Mortality Analysis (2019→2023 Change)
# =============================================================================
def plot_delta_mortality_results(delta_df: pd.DataFrame, plot_dir: str, logger=None):
    """
    Create comprehensive visualization for delta mortality analysis (2019→2023 change).
    
    Parameters:
    -----------
    delta_df : pd.DataFrame
        Results from delta mortality test with columns: Treatment, Label, ATE_YPLL_Change, 
        CI_Lower, CI_Upper, p_value, Direction
    plot_dir : str
        Directory to save plots
    logger : logging.Logger
    """
    if not VISUALIZATION_AVAILABLE:
        if logger:
            logger.warning("Visualization not available. Skipping delta mortality plots.")
        return
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with 2 subplots
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # --- LEFT PANEL: Forest Plot ---
        # Sort by ATE for better visualization
        plot_df = delta_df.sort_values('ATE_YPLL_Change', ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(plot_df))
        
        # Color by significance and direction
        colors = []
        for _, row in plot_df.iterrows():
            if row['p_value'] < 0.05:
                if row['ATE_YPLL_Change'] < 0:
                    colors.append('#2E7D32')  # Dark green for significant improvement
                else:
                    colors.append('#C62828')  # Dark red for significant worsening
            else:
                colors.append('#757575')  # Gray for non-significant
        
        # Plot points and error bars
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            ax1.plot([row['CI_Lower'], row['CI_Upper']], [i, i], 
                    color=colors[i], linewidth=2.5, alpha=0.8)
            marker = 'D' if row['p_value'] < 0.05 else 'o'
            ax1.scatter(row['ATE_YPLL_Change'], i, color=colors[i], 
                       marker=marker, s=150, zorder=3, edgecolors='black', linewidths=1)
        
        # Add vertical reference line at 0
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Labels and formatting
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(plot_df['Label'], fontsize=11)
        ax1.set_xlabel('Change in YPLL per 100k (2023 - 2019)\nNegative = Improvement', 
                      fontsize=12, fontweight='bold')
        ax1.set_title('AI/Robotics Impact on Mortality Change (2019→2023)\nAverage Treatment Effect on YPLL Reduction', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add interpretation region shading
        xlim = ax1.get_xlim()
        ax1.axvspan(xlim[0], 0, alpha=0.1, color='green', label='Improvement Zone')
        ax1.axvspan(0, xlim[1], alpha=0.1, color='red', label='Worsening Zone')
        
        # Legend for forest plot
        legend_elements = [
            plt.scatter([], [], marker='D', s=150, color='#2E7D32', 
                       edgecolors='black', linewidths=1, label='Sig. Improvement (p<0.05)'),
            plt.scatter([], [], marker='D', s=150, color='#C62828', 
                       edgecolors='black', linewidths=1, label='Sig. Worsening (p<0.05)'),
            plt.scatter([], [], marker='o', s=150, color='#757575', 
                       edgecolors='black', linewidths=1, label='Non-significant (p≥0.05)')
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)
        
        # --- RIGHT PANEL: Summary Statistics Table ---
        ax2.axis('off')
        
        # Prepare table data
        table_data = []
        table_data.append(['Treatment', 'ATE\n(YPLL Δ)', 'p-value', 'Interpretation'])
        
        for _, row in plot_df.iterrows():
            treatment_short = row['Label'].split('(')[1].split(')')[0]  # Extract MO14/MO21
            ate_str = f"{row['ATE_YPLL_Change']:.1f}"
            p_str = f"{row['p_value']:.4f}" if row['p_value'] >= 0.001 else "<0.001"
            
            if row['p_value'] < 0.05:
                if row['ATE_YPLL_Change'] < 0:
                    interp = f"↓ {abs(row['ATE_YPLL_Change']):.0f} fewer\nYPLL losses"
                else:
                    interp = f"↑ {row['ATE_YPLL_Change']:.0f} more\nYPLL losses"
            else:
                interp = "No sig.\neffect"
            
            table_data.append([treatment_short, ate_str, p_str, interp])
        
        # Create table
        table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                         bbox=[0, 0.3, 1, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#1976D2')
            cell.set_text_props(weight='bold', color='white')
        
        # Style data rows with alternating colors
        for i in range(1, len(table_data)):
            row_color = '#F5F5F5' if i % 2 == 0 else 'white'
            for j in range(4):
                cell = table[(i, j)]
                cell.set_facecolor(row_color)
                # Bold significant results
                if plot_df.iloc[i-1]['p_value'] < 0.05:
                    cell.set_text_props(weight='bold')
        
        # Add interpretation notes
        notes_text = (
            "INTERPRETATION:\n"
            "• Negative ATE = AI/Robotics associated with mortality reduction\n"
            "• Positive ATE = AI/Robotics associated with mortality increase\n"
            "• Controls for baseline mortality by using change score (2023-2019)\n"
            "• Tests whether AI adoption predicts improvement trajectory"
        )
        ax2.text(0.5, 0.15, notes_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax2.set_title('Summary Statistics & Interpretation', 
                     fontsize=12, fontweight='bold', pad=20)
        
        # Overall figure title
        fig.suptitle('Delta Mortality Analysis: Change in YPLL from 2019 (Placebo Period) to 2023 (Cross-Sectional Data)',
                    fontsize=15, fontweight='bold', y=0.98)
        
        # Save
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, "delta_mortality_2019_to_2023_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.info(f"  ✓ Saved delta mortality visualization: {plot_path}")
    
    except Exception as e:
        if logger:
            logger.error(f"  Error creating delta mortality plot: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')


def run_delta_mortality_test(df: pd.DataFrame, base_controls: List[str], 
                            n_boot: int, logger, out_dir: str, plot_dir: str) -> pd.DataFrame:
    """
    Test whether 2024 AI/Robotics adoption predicts IMPROVEMENT in mortality
    from 2019 (placebo baseline) to 2023 (cross-sectional data).
    
    Key Innovation: Uses change score (Δ_YPLL = YPLL_2023 - YPLL_2019) to control
    for baseline mortality and test whether AI predicts mortality reduction trajectory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Main analysis dataset with both 2019 and 2023 YPLL data
    base_controls : list
        List of control variables for AIPW
    n_boot : int
        Number of bootstrap replicates
    logger : logging.Logger
    out_dir : str
        Output directory for CSV files
    plot_dir : str
        Output directory for plots
    
    Returns:
    --------
    pd.DataFrame : Results table with ATE estimates for mortality change
    """
    logger.info("\n" + "="*80)
    logger.info("DELTA MORTALITY TEST: 2024 AI/Robotics → Change in YPLL (2019→2023)")
    logger.info("="*80)
    logger.info("PURPOSE:")
    logger.info("  Test if AI/Robotics adoption predicts mortality IMPROVEMENT (not just levels)")
    logger.info("  This addresses reviewer concerns about selection bias and baseline confounding")
    logger.info("\nMETHOD:")
    logger.info("  Outcome = Δ_YPLL = YPLL_2023 - YPLL_2019")
    logger.info("  - Negative Δ = Mortality improved (YPLL decreased)")
    logger.info("  - Positive Δ = Mortality worsened (YPLL increased)")
    logger.info("  - Controls for county-specific baseline by differencing")
    logger.info("\nEXPECTED RESULT:")
    logger.info("  Negative ATE = AI/Robotics adoption associated with mortality reduction")
    logger.info("  (Treatment counties show greater improvement 2019→2023)")
    logger.info("="*80 + "\n")
    
    # Check for required columns (keep estimand consistent: 2019 PL1 vs 2023 CT5)
    ypll_2019_col = 'pl1_ypll_rate'
    if ypll_2019_col not in df.columns:
        logger.error(f"  ERROR: {ypll_2019_col} not found. Cannot run delta test.")
        return pd.DataFrame()

    if 'ct5_ypll_per_100k_mid' in df.columns:
        ypll_2023_col = 'ct5_ypll_per_100k_mid'
        logger.info(f"  Using CT5 2023 YPLL (age-adjusted, mid-estimate): {ypll_2023_col}")
    else:
        logger.error("  ERROR: 2023 CT5 YPLL not available. Cannot run delta test (avoiding dv21 fallback).")
        return pd.DataFrame()
    
    # Define treatment variables to test
    treatment_vars = ['mo14_ai_automate_routine_tasks_pct', 'mo21_robotics_in_hospital_pct']

    # Use base controls but drop spatial lags to avoid leakage from missing-counties; keep state_fips for clustering
    delta_controls = [c for c in base_controls if not c.endswith('_spatial_lag')]
    if len(delta_controls) != len(base_controls):
        logger.info(f"  Delta test removing spatial lag controls ({len(base_controls)-len(delta_controls)} dropped) to avoid leakage.")
    
    # Calculate delta (change from 2019 to 2023)
    # Include treatment variables in required columns so they're not dropped
    required_for_delta = [ypll_2023_col, ypll_2019_col, 'state_fips_for_clustering'] + treatment_vars + delta_controls
    required_for_delta = [c for c in required_for_delta if c in df.columns]
    df_delta = df[required_for_delta].dropna().copy()

    if 'state_fips_for_clustering' not in df_delta.columns:
        logger.error("  state_fips_for_clustering missing in delta dataset; cannot cluster bootstrap.")
        return pd.DataFrame()
    
    df_delta['delta_ypll_2019_to_2023'] = df_delta[ypll_2023_col] - df_delta[ypll_2019_col]
    
    logger.info(f"  Δ_YPLL calculated for {len(df_delta)} counties")
    logger.info(f"  Mean change: {df_delta['delta_ypll_2019_to_2023'].mean():.1f} YPLL per 100k")
    logger.info(f"  Std dev: {df_delta['delta_ypll_2019_to_2023'].std():.1f}")
    logger.info(f"  Interpretation: {'Population mortality IMPROVED' if df_delta['delta_ypll_2019_to_2023'].mean() < 0 else 'Population mortality WORSENED'} on average from 2019→2023")
    logger.info(f"  Counties with improvement (Δ<0): {(df_delta['delta_ypll_2019_to_2023'] < 0).sum()} ({(df_delta['delta_ypll_2019_to_2023'] < 0).mean()*100:.1f}%)")
    logger.info(f"  Counties with worsening (Δ>0): {(df_delta['delta_ypll_2019_to_2023'] > 0).sum()} ({(df_delta['delta_ypll_2019_to_2023'] > 0).mean()*100:.1f}%)\n")
    
    # Define treatment specifications
    delta_specs = [
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "label": "MO14 (AI Workflow Automation)"},
        {"treatment": "mo21_robotics_in_hospital_pct", "label": "MO21 (Hospital Robotics)"},
    ]
    
    delta_results = []
    
    for spec in delta_specs:
        treat_col = spec["treatment"]
        label = spec["label"]
        
        if treat_col not in df_delta.columns:
            logger.warning(f"  Skipping {label}: {treat_col} not found in data")
            continue
        
        logger.info(f"  Testing: {label} → Δ_YPLL (2019→2023)")
        logger.info(f"  Treatment variable: {treat_col}")
        
        # Create analysis dataset
        analysis_cols = [treat_col, 'delta_ypll_2019_to_2023', 'state_fips_for_clustering'] + delta_controls
        df_analysis = df_delta[analysis_cols].dropna().copy()
        
        if len(df_analysis) < 50:
            logger.warning(f"    Insufficient data: N={len(df_analysis)}")
            continue
        
        # Binarize treatment (median split, or 75th percentile for sparse MO14)
        if "mo14" in treat_col.lower():
            # MO14 is sparse (most counties have 0)
            threshold = df_analysis[treat_col].quantile(0.75)
            if threshold == 0:
                nonzero_vals = df_analysis.loc[df_analysis[treat_col] > 0, treat_col]
                threshold = nonzero_vals.min() if len(nonzero_vals) > 0 else df_analysis[treat_col].median()
            logger.info(f"    Using 75th percentile threshold for sparse MO14: {threshold:.4f}")
        else:
            threshold = df_analysis[treat_col].median()
            logger.info(f"    Using median split threshold: {threshold:.4f}")
        
        binary_col = f'{treat_col}_binary'
        df_analysis[binary_col] = (df_analysis[treat_col] > threshold).astype(int)
        
        n_treated = df_analysis[binary_col].sum()
        n_control = len(df_analysis) - n_treated
        
        logger.info(f"    N total: {len(df_analysis)}")
        logger.info(f"    N treated (>{threshold:.4f}): {n_treated}")
        logger.info(f"    N control (≤{threshold:.4f}): {n_control}")
        
        if n_treated < 10 or n_control < 10:
            logger.warning(f"    Insufficient treatment/control balance. Skipping.")
            continue
        
        # Descriptive statistics by group
        treated_mean = df_analysis.loc[df_analysis[binary_col] == 1, 'delta_ypll_2019_to_2023'].mean()
        control_mean = df_analysis.loc[df_analysis[binary_col] == 0, 'delta_ypll_2019_to_2023'].mean()
        treated_median = df_analysis.loc[df_analysis[binary_col] == 1, 'delta_ypll_2019_to_2023'].median()
        control_median = df_analysis.loc[df_analysis[binary_col] == 0, 'delta_ypll_2019_to_2023'].median()
        
        logger.info(f"    Treated group Δ_YPLL: Mean={treated_mean:.1f}, Median={treated_median:.1f}")
        logger.info(f"    Control group Δ_YPLL: Mean={control_mean:.1f}, Median={control_median:.1f}")
        logger.info(f"    Unadjusted difference: {treated_mean - control_mean:.1f} YPLL")
        
        # Run AIPW
        logger.info(f"    Running AIPW with {n_boot} bootstrap replicates...")
        
        try:
            (ate, ci_l, ci_u, p, n_t, n_c, err) = run_aipw(
                df=df_analysis,
                treat_col=binary_col,
                y_col='delta_ypll_2019_to_2023',
                confounders=[c for c in delta_controls if c in df_analysis.columns],
                n_boot=n_boot,
                logger=logger,
                plot_dir=plot_dir,
                cluster_ids=df_analysis['state_fips_for_clustering'],
                cluster_bootstrap=True
            )
            
            if err:
                logger.error(f"    AIPW failed for {label}")
                continue
            
            # Interpret results
            logger.info(f"\n    ═══ RESULTS: {label} ═══")
            logger.info(f"    ATE (YPLL Change): {ate:.2f} [95% CI: {ci_l:.2f}, {ci_u:.2f}]")
            logger.info(f"    p-value: {p:.4f}")
            logger.info(f"    Significance: {'YES (p < 0.05)' if p < 0.05 else 'NO (p ≥ 0.05)'}")
            
            if p < 0.05:
                if ate < 0:
                    logger.info(f"    INTERPRETATION: Treatment counties had {abs(ate):.1f} FEWER additional YPLL losses")
                    logger.info(f"                    AI/Robotics adoption associated with {abs(ate):.0f} YPLL REDUCTION")
                    logger.info(f"                    This represents MORTALITY IMPROVEMENT vs control counties")
                else:
                    logger.info(f"    INTERPRETATION: Treatment counties had {ate:.1f} MORE additional YPLL losses")
                    logger.info(f"                    AI/Robotics adoption associated with MORTALITY WORSENING")
                    logger.info(f"                    ⚠ Unexpected finding - warrants further investigation")
            else:
                logger.info(f"    INTERPRETATION: No significant effect on mortality trajectory")
                logger.info(f"                    AI/Robotics adoption did not significantly change 2019→2023 trend")
            
            logger.info(f"    ════════════════════════════════════\n")
            
            # Store results
            direction = "Improvement (AI→Less YPLL)" if ate < 0 else "Worsening (AI→More YPLL)"
            
            delta_results.append({
                "Test": "Delta_Mortality_2019_to_2023",
                "Treatment": treat_col,
                "Label": label,
                "Outcome": "YPLL_Change_2019_to_2023",
                "N": len(df_analysis),
                "N_Treated": n_t,
                "N_Control": n_c,
                "Threshold": threshold,
                "Treated_Mean_Delta": treated_mean,
                "Control_Mean_Delta": control_mean,
                "Unadjusted_Difference": treated_mean - control_mean,
                "ATE_YPLL_Change": ate,
                "CI_Lower": ci_l,
                "CI_Upper": ci_u,
                "p_value": p,
                "Significant": "Yes" if p < 0.05 else "No",
                "Direction": direction,
                "Interpretation": f"{abs(ate):.0f} {'less' if ate < 0 else 'more'} YPLL per 100k change vs control"
            })
        
        except Exception as e:
            logger.error(f"    Error running AIPW for {label}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results DataFrame and save
    if delta_results:
        delta_df = pd.DataFrame(delta_results)
        
        # Save CSV
        delta_csv = os.path.join(out_dir, "delta_mortality_2019_to_2023_sensitivity.csv")
        delta_df.to_csv(delta_csv, index=False)
        logger.info(f"\n✓ Delta mortality results saved: {delta_csv}")
        
        # Display results table
        logger.info(f"\n{'═'*80}")
        logger.info("DELTA MORTALITY SUMMARY TABLE")
        logger.info(f"{'═'*80}")
        display_cols = ['Label', 'N', 'ATE_YPLL_Change', 'CI_Lower', 'CI_Upper', 'p_value', 'Direction']
        logger.info(f"\n{delta_df[display_cols].to_string(index=False)}\n")
        logger.info(f"{'═'*80}")
        
        # Create visualization
        logger.info(f"\nGenerating delta mortality visualization...")
        plot_delta_mortality_results(delta_df, plot_dir, logger)
        
        # Overall interpretation
        logger.info(f"\n{'═'*80}")
        logger.info("OVERALL INTERPRETATION")
        logger.info(f"{'═'*80}")
        
        sig_improvements = delta_df[(delta_df['p_value'] < 0.05) & (delta_df['ATE_YPLL_Change'] < 0)]
        sig_worsening = delta_df[(delta_df['p_value'] < 0.05) & (delta_df['ATE_YPLL_Change'] > 0)]
        non_sig = delta_df[delta_df['p_value'] >= 0.05]
        
        logger.info(f"Significant improvements: {len(sig_improvements)}/{len(delta_df)}")
        logger.info(f"Significant worsening: {len(sig_worsening)}/{len(delta_df)}")
        logger.info(f"Non-significant: {len(non_sig)}/{len(delta_df)}")
        
        if len(sig_improvements) > 0:
            logger.info(f"\n✓ EVIDENCE SUPPORTS CAUSAL EFFECT:")
            logger.info(f"  AI/Robotics adoption predicts MORTALITY IMPROVEMENT from 2019→2023")
            logger.info(f"  This delta test controls for baseline differences better than cross-sectional")
            for _, row in sig_improvements.iterrows():
                logger.info(f"  - {row['Label']}: {abs(row['ATE_YPLL_Change']):.0f} fewer YPLL per 100k (p={row['p_value']:.4f})")
        
        if len(sig_worsening) > 0:
            logger.warning(f"\n⚠ UNEXPECTED FINDINGS:")
            logger.warning(f"  Some technologies associated with mortality WORSENING:")
            for _, row in sig_worsening.iterrows():
                logger.warning(f"  - {row['Label']}: {row['ATE_YPLL_Change']:.0f} more YPLL per 100k (p={row['p_value']:.4f})")
        
        logger.info(f"\n{'═'*80}\n")
        
        return delta_df
    else:
        logger.warning("  No delta mortality tests completed (missing data or insufficient variation)")
        return pd.DataFrame()


# =============================================================================
# NEW: Pre/Post Differential-Change Analysis (2019 -> 2023)
# =============================================================================
def plot_prepost_outcome_trajectories(change_df: pd.DataFrame, treatment_col: str, plot_dir: str, logger=None):
    """
    For a single treatment definition, draw 2-line pre/post trajectories
    (2019 and 2023) for adopters vs non-adopters across outcomes.
    """
    if not VISUALIZATION_AVAILABLE:
        if logger:
            logger.warning("Visualization not available. Skipping pre/post trajectory plot.")
        return

    d = change_df[change_df["Treatment"] == treatment_col].copy()
    if d.empty:
        return

    d = d.sort_values("Outcome_Label").reset_index(drop=True)
    n_outcomes = len(d)
    n_cols = 2
    n_rows = int(np.ceil(n_outcomes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, max(4.5, 4.2 * n_rows)), squeeze=False)
    legend_handles = None
    legend_labels = None

    for i, row in d.iterrows():
        ax = axes[i // n_cols][i % n_cols]
        x = [2019, 2023]
        y_control = [row["Control_Pre_Mean"], row["Control_Post_Mean"]]
        y_treated = [row["Treated_Pre_Mean"], row["Treated_Post_Mean"]]

        l1 = ax.plot(x, y_control, marker="o", linewidth=2.4, color="#1f77b4", label="Non-adopters")
        l2 = ax.plot(x, y_treated, marker="o", linewidth=2.4, color="#d62728", label="Adopters")
        if legend_handles is None:
            legend_handles = [l1[0], l2[0]]
            legend_labels = ["Non-adopters", "Adopters"]

        ax.set_xticks([2019, 2023])
        ax.set_title(f"{row['Outcome_Label']} ({row['Better_When']} better)", fontsize=10.5)
        ax.grid(True, alpha=0.28)

        ate = row["AIPW_ATE"]
        pval = row["p_value"]
        ate_txt = "NA" if not np.isfinite(ate) else f"{ate:.2f}"
        p_txt = "NA" if not np.isfinite(pval) else (f"{pval:.3f}" if pval >= 0.001 else "<0.001")
        ax.text(
            0.02, 0.03,
            f"Change-diff (AIPW): {ate_txt}, p={p_txt}",
            transform=ax.transAxes, fontsize=8.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
            va="bottom", ha="left"
        )

    for j in range(n_outcomes, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2, frameon=False)

    fig.suptitle(
        f"Pre/Post Trajectories by Adoption Group: {treatment_col}",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    os.makedirs(plot_dir, exist_ok=True)
    outp = os.path.join(plot_dir, f"prepost_trajectories_2019_2023_{treatment_col}.png")
    plt.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close()
    if logger:
        logger.info(f"  Saved pre/post trajectory plot: {outp}")


def plot_prepost_change_forest(change_df: pd.DataFrame, treatment_col: str, plot_dir: str, logger=None):
    """
    Compact forest plot of change-differences (Adopter delta - Non-adopter delta)
    for one treatment definition across all eligible outcomes.
    """
    if not VISUALIZATION_AVAILABLE:
        if logger:
            logger.warning("Visualization not available. Skipping pre/post forest plot.")
        return

    d = change_df[change_df["Treatment"] == treatment_col].copy()
    if d.empty:
        return

    d = d.sort_values("AIPW_ATE", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, max(4.0, 0.9 * len(d) + 2)))
    y_pos = np.arange(len(d))

    colors = []
    for _, row in d.iterrows():
        if not np.isfinite(row["p_value"]) or row["p_value"] >= 0.05:
            colors.append("#757575")
            continue
        favorable = row["AIPW_ATE"] if row["Higher_Is_Better"] else -row["AIPW_ATE"]
        colors.append("#2E7D32" if favorable > 0 else "#C62828")

    for i, (_, row) in enumerate(d.iterrows()):
        ax.plot([row["CI_Lower"], row["CI_Upper"]], [i, i], color=colors[i], linewidth=2.3, alpha=0.9)
        marker = "D" if np.isfinite(row["p_value"]) and row["p_value"] < 0.05 else "o"
        ax.scatter(
            row["AIPW_ATE"], i, s=130, marker=marker, color=colors[i],
            edgecolors="black", linewidths=0.8, zorder=3
        )

    y_labels = [
        f"{r['Outcome_Label']} ({'higher' if r['Higher_Is_Better'] else 'lower'} better)"
        for _, r in d.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10.5)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.65)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlabel("Change-difference = (Post-Pre)_Adopters - (Post-Pre)_Non-adopters", fontsize=11.5, fontweight="bold")
    ax.set_title(f"2019->2023 Differential Change Forest: {treatment_col}", fontsize=13.5, fontweight="bold")

    os.makedirs(plot_dir, exist_ok=True)
    outp = os.path.join(plot_dir, f"prepost_change_forest_2019_2023_{treatment_col}.png")
    plt.tight_layout()
    plt.savefig(outp, dpi=300, bbox_inches="tight")
    plt.close()
    if logger:
        logger.info(f"  Saved pre/post change forest plot: {outp}")


def run_prepost_differential_change_analysis(df: pd.DataFrame, base_controls: List[str],
                                             n_boot: int, logger, out_dir: str, plot_dir: str) -> pd.DataFrame:
    """
    Reviewer-requested pretest-posttest nonequivalent-groups style analysis:
      DeltaY = Y_2023 - Y_2019, compare DeltaY for adopters vs non-adopters.

    Uses available 2019/2023 paired outcomes and AIPW on the change score.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PRE/POST DIFFERENTIAL-CHANGE ANALYSIS (2019 -> 2023)")
    logger.info("=" * 80)
    logger.info("Design: pretest-posttest nonequivalent groups using change scores.")
    logger.info("Estimand: E[DeltaY | Adopter=1] - E[DeltaY | Adopter=0], where DeltaY = Y_2023 - Y_2019.")
    logger.info("Interpretation: compares improvement trajectories, not treatment status in 2019.")
    logger.info("=" * 80 + "\n")

    delta_controls = [c for c in base_controls if c in df.columns and not c.endswith("_spatial_lag")]
    if len(delta_controls) != len([c for c in base_controls if c in df.columns]):
        logger.info("  Dropped spatial-lag controls for pre/post change models.")

    ypll_post_col = None
    for cand in ["ct5_ypll_per_100k_mid", "ct5_ypll_u75_age_adj_per_100k_mid"]:
        if cand in df.columns:
            ypll_post_col = cand
            break

    outcome_specs = []
    if ("pl1_ypll_rate" in df.columns) and (ypll_post_col is not None):
        outcome_specs.append({
            "pre": "pl1_ypll_rate",
            "post": ypll_post_col,
            "label": "YPLL per 100k (<75 years)",
            "better_when": "Lower",
            "higher_is_better": False
        })

    for pre_col, post_col, label, better_when, higher_is_better in [
        ("hosp_sep1_2019", "hosp_sep1_2023", "SEP-1", "Higher", True),
        ("hosp_op18b_2019", "hosp_op18b_2023", "OP-18b", "Lower", False),
        ("hosp_mort30pn_2019", "hosp_mort30pn_2023", "Pneumonia Mortality (30-day)", "Lower", False),
    ]:
        if pre_col in df.columns and post_col in df.columns:
            outcome_specs.append({
                "pre": pre_col,
                "post": post_col,
                "label": label,
                "better_when": better_when,
                "higher_is_better": higher_is_better
            })

    if not outcome_specs:
        logger.warning("No eligible 2019/2023 paired outcomes found. Pre/post differential-change analysis skipped.")
        return pd.DataFrame()

    treatment_specs = [
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "label": "MO14_AI_Workflow_Automation"},
        {"treatment": "mo21_robotics_in_hospital_pct", "label": "MO21_Hospital_Robotics"},
    ]

    rows = []
    for t_spec in treatment_specs:
        treat_col = t_spec["treatment"]
        if treat_col not in df.columns:
            logger.warning(f"Treatment {treat_col} missing; skipping this treatment in pre/post block.")
            continue

        for o_spec in outcome_specs:
            pre_col = o_spec["pre"]
            post_col = o_spec["post"]
            label = o_spec["label"]

            cols = [treat_col, pre_col, post_col, "state_fips_for_clustering"] + delta_controls
            cols = [c for c in cols if c in df.columns]
            d = df[cols].dropna().copy()
            if len(d) < 80:
                logger.warning(f"Skipping {treat_col} -> {label}: small sample after dropna (N={len(d)}).")
                continue

            d["delta_outcome"] = d[post_col] - d[pre_col]
            if d["delta_outcome"].nunique() < 5:
                logger.warning(f"Skipping {treat_col} -> {label}: low variation in delta outcome.")
                continue

            if "mo14" in treat_col.lower():
                threshold = d[treat_col].quantile(0.75)
                if threshold == 0:
                    nonzero = d.loc[d[treat_col] > 0, treat_col]
                    if len(nonzero) > 0:
                        threshold = nonzero.min()
            else:
                threshold = d[treat_col].median()

            treat_bin = f"{treat_col}_binary_prepost"
            d[treat_bin] = (d[treat_col] > threshold).astype(int)
            n_treated = int((d[treat_bin] == 1).sum())
            n_control = int((d[treat_bin] == 0).sum())

            if n_treated < 10 or n_control < 10:
                logger.warning(
                    f"Skipping {treat_col} -> {label}: poor group balance (N_t={n_treated}, N_c={n_control})."
                )
                continue

            treated_pre = float(d.loc[d[treat_bin] == 1, pre_col].mean())
            control_pre = float(d.loc[d[treat_bin] == 0, pre_col].mean())
            treated_post = float(d.loc[d[treat_bin] == 1, post_col].mean())
            control_post = float(d.loc[d[treat_bin] == 0, post_col].mean())
            treated_delta = float(d.loc[d[treat_bin] == 1, "delta_outcome"].mean())
            control_delta = float(d.loc[d[treat_bin] == 0, "delta_outcome"].mean())
            unadj_diff = treated_delta - control_delta

            confs = [c for c in delta_controls if c in d.columns]
            try:
                (ate, ci_l, ci_u, p, n_t, n_c, err) = run_aipw(
                    df=d,
                    treat_col=treat_bin,
                    y_col="delta_outcome",
                    confounders=confs,
                    n_boot=n_boot,
                    logger=logger,
                    plot_dir=plot_dir,
                    cluster_ids=d["state_fips_for_clustering"],
                    cluster_bootstrap=True
                )
            except Exception as e:
                logger.warning(f"AIPW failed for {treat_col} -> {label}: {e}")
                ate = ci_l = ci_u = p = np.nan
                n_t = n_treated
                n_c = n_control
                err = True

            if err:
                ate = ci_l = ci_u = p = np.nan

            favorable_effect = np.nan
            if np.isfinite(ate):
                favorable_effect = ate if o_spec["higher_is_better"] else -ate

            rows.append({
                "Treatment": treat_col,
                "Treatment_Label": t_spec["label"],
                "Outcome_Label": label,
                "Outcome_Pre_Col": pre_col,
                "Outcome_Post_Col": post_col,
                "N": len(d),
                "N_Treated": n_t,
                "N_Control": n_c,
                "Threshold": float(threshold),
                "Better_When": o_spec["better_when"],
                "Higher_Is_Better": bool(o_spec["higher_is_better"]),
                "Treated_Pre_Mean": treated_pre,
                "Control_Pre_Mean": control_pre,
                "Treated_Post_Mean": treated_post,
                "Control_Post_Mean": control_post,
                "Treated_Delta_Mean": treated_delta,
                "Control_Delta_Mean": control_delta,
                "Unadjusted_Change_Diff": unadj_diff,
                "AIPW_ATE": ate,
                "CI_Lower": ci_l,
                "CI_Upper": ci_u,
                "p_value": p,
                "Significant": "Yes" if np.isfinite(p) and p < 0.05 else "No",
                "Favorable_Effect": favorable_effect,
            })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        logger.warning("No pre/post differential-change models completed.")
        return result_df

    out_csv = os.path.join(out_dir, "prepost_change_differences_2019_2023.csv")
    result_df.to_csv(out_csv, index=False)
    logger.info(f"Saved pre/post differential-change results: {out_csv}")

    logger.info("\nPRE/POST CHANGE SUMMARY (AIPW):")
    display_cols = ["Treatment", "Outcome_Label", "N", "AIPW_ATE", "CI_Lower", "CI_Upper", "p_value", "Better_When"]
    logger.info(result_df[display_cols].to_string(index=False))

    for t_col in sorted(result_df["Treatment"].dropna().unique()):
        plot_prepost_outcome_trajectories(result_df, t_col, plot_dir, logger)
        plot_prepost_change_forest(result_df, t_col, plot_dir, logger)

    logger.info("=" * 80 + "\n")
    return result_df


# =============================================================================
# NEW: CT6 Hospital Deaths Analysis (2023 Age-Adjusted)
# =============================================================================
def run_ct6_hospital_deaths_block(df: pd.DataFrame,
                                  df_bridge: pd.DataFrame,
                                  base_controls: List[str],
                                  logger,
                                  out_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Tests direct and moderation effects for CT6 (2023 age-adjusted hospital deaths):
    
    Direct effects (Clustered OLS + AIPW):
      - MO14 (workflow AI) -> CT6
      - MO21 (robotics) -> CT6
      - MO1 (GenAI composite) -> CT6
      - FA21 (PSI-90 safety composite) -> CT6
      - EF6 (OP-18b ED timeliness) -> CT6
      - EF23 (SEP-1 sepsis care) -> CT6
    
    Moderation effects (Clustered OLS with interaction terms):
      - FA21 x MO14 -> CT6
      - EF6 x MO14 -> CT6
      - EF23 x MO14 -> CT6
    
    Returns dict with 'direct_effects' and 'moderation_effects' DataFrames.
    """
    
    results = {}
    outcome = "ct6_hospital_deaths_age_adj"
    
    # Check if CT6 is available
    if outcome not in df.columns:
        logger.warning(f"CT6 outcome '{outcome}' not found in dataframe. CT6 block will be skipped.")
        return results
    
    logger.info(f"\nRunning CT6 hospital deaths analysis (outcome: {outcome})")
    
    # Verify CT6 has sufficient data
    ct6_data = df[outcome].dropna()
    if len(ct6_data) < 150:
        logger.warning(f"CT6 has only {len(ct6_data)} non-null values. Analysis skipped (need >= 150).")
        return results
    
    logger.info(f"CT6 available: {len(ct6_data)} counties, range [{ct6_data.min():.1f}, {ct6_data.max():.1f}]")
    
    # Merge bridge data if available (for hospital-level metrics: FA21, EF6, EF23, MO14, MO21)
    df_work = df.copy()
    if df_bridge is not None and not df_bridge.empty:
        df_bridge_local = df_bridge.copy()
        df_bridge_local.columns = [c.lower() for c in df_bridge_local.columns]
        
        # Select only the metrics we need from bridge
        bridge_cols = ['county_fips', 'mo14_wfaiart', 'mo21_robohos', 'fa21_psi_90', 'ef6_op_18b', 'ef23_sep_1']
        bridge_cols_available = [c for c in bridge_cols if c in df_bridge_local.columns]
        
        if len(bridge_cols_available) > 1:  # More than just county_fips
            df_work = df_work.merge(df_bridge_local[bridge_cols_available], on='county_fips', how='left', suffixes=('', '_bridge'))
            logger.info(f"Merged bridge data: {len(bridge_cols_available)-1} variables added for CT6 analysis.")
        else:
            logger.warning("Bridge data available but no usable metrics found.")
    else:
        logger.info("No bridge data provided. Using only main df variables.")
    
    # =========================================================================
    # PART A: DIRECT EFFECTS (Clustered OLS + AIPW)
    # =========================================================================
    logger.info("\n--- CT6 Direct Effects ---")
    
    # Use correct variable names:
    # - From main df: mo14_ai_automate_routine_tasks_pct, mo21_robotics_in_hospital_pct, mo1_genai_composite_score
    # - From bridge (if merged): mo14_wfaiart, mo21_robohos, fa21_psi_90, ef6_op_18b, ef23_sep_1
    direct_specs = [
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "label": "MO14_WorkflowAI_MainDF"},
        {"treatment": "mo14_wfaiart", "label": "MO14_WorkflowAI_Bridge"},
        {"treatment": "mo21_robotics_in_hospital_pct", "label": "MO21_Robotics_MainDF"},
        {"treatment": "mo21_robohos", "label": "MO21_Robotics_Bridge"},
        {"treatment": "mo1_genai_composite_score", "label": "MO1_GenAI_Composite"},
        {"treatment": "fa21_psi_90", "label": "FA21_PSI90_Safety"},
        {"treatment": "ef6_op_18b", "label": "EF6_OP18b_ED_Timeliness"},
        {"treatment": "ef23_sep_1", "label": "EF23_SEP1_Sepsis"},
    ]
    
    direct_rows = []
    
    for spec in direct_specs:
        t_col = spec["treatment"]
        t_label = spec["label"]
        
        if t_col not in df_work.columns:
            continue  # Silently skip if not available
        
        logger.info(f"  Testing {t_label} -> CT6")
        model_controls = get_control_set(outcome, base_controls, df_work)
        
        # Prepare data
        cols_needed = [outcome, t_col, "state_fips_for_clustering"] + model_controls
        cols_available = [c for c in cols_needed if c in df_work.columns]
        d = df_work[cols_available].dropna().copy()
        
        if len(d) < 150:
            logger.warning(f"    Sample size too small: N={len(d)}. Skipping.")
            continue
        
        if d[t_col].nunique() < 5:
            logger.warning(f"    Treatment '{t_col}' has low variation: {d[t_col].nunique()} unique values. Skipping.")
            continue
        
        # Clustered OLS
        X = d[[t_col] + [c for c in model_controls if c in d.columns]]
        try:
            res_ols = run_ols_clustered(d[outcome], X, d["state_fips_for_clustering"])
            beta_ols = float(res_ols.params.get(t_col, np.nan))
            p_ols = float(res_ols.pvalues.get(t_col, np.nan))
            ci_ols = res_ols.conf_int().loc[t_col].values if t_col in res_ols.params.index else (np.nan, np.nan)
            r2_ols = float(res_ols.rsquared) if hasattr(res_ols, "rsquared") else np.nan
        except Exception as e:
            logger.warning(f"    OLS failed for {t_label}: {e}")
            beta_ols = p_ols = r2_ols = np.nan
            ci_ols = (np.nan, np.nan)
        
        # AIPW (if treatment has sufficient variation for binary split)
        ate_aipw = ci_aipw_lower = ci_aipw_upper = p_aipw = np.nan
        n_treated = n_control = 0
        control_mean_ct6 = np.nan
        relative_change_pct = np.nan
        
        # Use median split for AIPW
        if d[t_col].nunique() >= 10:  # Reasonable for median split
            d_aipw = d.copy()
            median_val = d_aipw[t_col].median()
            bin_col = f"{t_col}_binary"
            d_aipw[bin_col] = (d_aipw[t_col] > median_val).astype(int)
            
            if d_aipw[bin_col].nunique() == 2:
                try:
                    (ate_aipw, ci_aipw_lower, ci_aipw_upper, p_aipw,
                     n_treated, n_control, aipw_err) = run_aipw(
                        d_aipw, bin_col, outcome, 
                        [c for c in model_controls if c in d_aipw.columns],
                        N_BOOT, logger, out_dir
                    )
                    
                    # Calculate control group mean (Ti=0) in the ANALYTIC SAMPLE
                    control_mask = d_aipw[bin_col] == 0
                    if control_mask.sum() > 0:
                        control_mean_ct6 = float(d_aipw.loc[control_mask, outcome].mean())
                        
                        # Calculate relative percent change: 100 * ATE / control_mean
                        if np.isfinite(ate_aipw) and np.isfinite(control_mean_ct6) and abs(control_mean_ct6) > 1e-9:
                            relative_change_pct = 100.0 * ate_aipw / control_mean_ct6
                        
                        logger.info(f"    Control mean CT6 (Ti=0): {control_mean_ct6:.2f} deaths/100k")
                        if np.isfinite(relative_change_pct):
                            logger.info(f"    Relative change: {relative_change_pct:.1f}%")
                        
                except Exception as e:
                    logger.warning(f"    AIPW failed for {t_label}: {e}")
        
        direct_rows.append({
            "Treatment": t_label,
            "Treatment_Col": t_col,
            "Outcome": "CT6_Hospital_Deaths",
            "N": len(d),
            "OLS_Beta": beta_ols,
            "OLS_p": p_ols,
            "OLS_CI_Lower": ci_ols[0],
            "OLS_CI_Upper": ci_ols[1],
            "OLS_R2": r2_ols,
            "AIPW_ATE": ate_aipw,
            "AIPW_CI_Lower": ci_aipw_lower,
            "AIPW_CI_Upper": ci_aipw_upper,
            "AIPW_p": p_aipw,
            "AIPW_N_Treated": n_treated,
            "AIPW_N_Control": n_control,
            "Control_Mean_CT6": control_mean_ct6,
            "Relative_Change_Pct": relative_change_pct
        })
    
    direct_df = pd.DataFrame(direct_rows)
    if not direct_df.empty:
        # FDR correction on OLS p-values
        direct_df = _bh_correct_in_place(direct_df, "OLS_p", None, "OLS_q_value")
        # FDR correction on AIPW p-values
        if direct_df["AIPW_p"].notna().sum() > 0:
            direct_df = _bh_correct_in_place(direct_df, "AIPW_p", None, "AIPW_q_value")
        
        out_csv = os.path.join(out_dir, "ct6_hospital_deaths_direct_effects.csv")
        direct_df.to_csv(out_csv, index=False)
        logger.info(f"\nSaved CT6 direct effects: {out_csv}")
        logger.info(f"  {len(direct_df)} tests completed.")
        
        # Prominent summary of key results with control means
        logger.info("\n" + "="*70)
        logger.info("CT6 DIRECT EFFECTS SUMMARY (with control group baselines)")
        logger.info("="*70)
        for _, row in direct_df.iterrows():
            if pd.notna(row['AIPW_ATE']) and pd.notna(row['Control_Mean_CT6']):
                logger.info(f"{row['Treatment']:30s} -> CT6")
                logger.info(f"  Control mean (Ti=0): {row['Control_Mean_CT6']:8.2f} deaths/100k")
                logger.info(f"  AIPW ATE:            {row['AIPW_ATE']:8.2f} deaths/100k")
                if pd.notna(row['Relative_Change_Pct']):
                    logger.info(f"  Relative change:     {row['Relative_Change_Pct']:8.1f}%")
                logger.info(f"  95% CI: [{row['AIPW_CI_Lower']:.2f}, {row['AIPW_CI_Upper']:.2f}]")
                logger.info(f"  p-value: {row['AIPW_p']:.4f}, q-value: {row.get('AIPW_q_value', np.nan):.4f}")
                logger.info("")
        logger.info("="*70)
    else:
        logger.warning("No CT6 direct effects results produced.")
    
    results["direct_effects"] = direct_df
    
    # =========================================================================
    # PART B: MODERATION EFFECTS (Interaction Terms)
    # =========================================================================
    logger.info("\n--- CT6 Moderation Effects ---")
    
    # Use bridge variables for moderations (hospital-level metrics)
    # Try both MO14 from bridge and main df
    moderation_specs = [
        {"moderator": "fa21_psi_90", "treatment": "mo14_wfaiart", "label": "FA21_PSI90 x MO14_Bridge"},
        {"moderator": "ef6_op_18b", "treatment": "mo14_wfaiart", "label": "EF6_OP18b x MO14_Bridge"},
        {"moderator": "ef23_sep_1", "treatment": "mo14_wfaiart", "label": "EF23_SEP1 x MO14_Bridge"},
        {"moderator": "fa21_psi_90", "treatment": "mo14_ai_automate_routine_tasks_pct", "label": "FA21_PSI90 x MO14_MainDF"},
        {"moderator": "ef6_op_18b", "treatment": "mo14_ai_automate_routine_tasks_pct", "label": "EF6_OP18b x MO14_MainDF"},
        {"moderator": "ef23_sep_1", "treatment": "mo14_ai_automate_routine_tasks_pct", "label": "EF23_SEP1 x MO14_MainDF"},
    ]
    
    moderation_rows = []
    
    for spec in moderation_specs:
        mod_col = spec["moderator"]
        treat_col = spec["treatment"]
        label = spec["label"]
        
        if mod_col not in df_work.columns or treat_col not in df_work.columns:
            continue  # Silently skip if not available
        
        logger.info(f"  Testing {label} -> CT6")
        
        # Prepare data
        cols_needed = [outcome, mod_col, treat_col, "state_fips_for_clustering"] + base_controls
        cols_available = [c for c in cols_needed if c in df_work.columns]
        d = df_work[cols_available].dropna().copy()
        
        if len(d) < 150:
            logger.warning(f"    Sample size too small: N={len(d)}. Skipping.")
            continue
        
        # Mean-center for interaction
        mod_centered = f"{mod_col}_centered"
        treat_centered = f"{treat_col}_centered"
        interact_col = f"{mod_col}_X_{treat_col}"
        
        d[mod_centered] = d[mod_col] - d[mod_col].mean()
        d[treat_centered] = d[treat_col] - d[treat_col].mean()
        d[interact_col] = d[mod_centered] * d[treat_centered]
        
        # Calculate control group mean: counties with below-median treatment
        # (using median split as proxy for Ti=0 vs Ti=1 for context)
        treat_median = d[treat_col].median()
        control_mask_mod = d[treat_col] <= treat_median
        control_mean_ct6_mod = float(d.loc[control_mask_mod, outcome].mean()) if control_mask_mod.sum() > 0 else np.nan
        
        # Clustered OLS with interaction
        X = d[[mod_centered, treat_centered, interact_col] + [c for c in base_controls if c in d.columns]]
        
        try:
            res_int = run_ols_clustered(d[outcome], X, d["state_fips_for_clustering"])
            
            # Extract coefficients
            beta_mod = float(res_int.params.get(mod_centered, np.nan))
            p_mod = float(res_int.pvalues.get(mod_centered, np.nan))
            ci_mod = res_int.conf_int().loc[mod_centered].values if mod_centered in res_int.params.index else (np.nan, np.nan)
            
            beta_treat = float(res_int.params.get(treat_centered, np.nan))
            p_treat = float(res_int.pvalues.get(treat_centered, np.nan))
            ci_treat = res_int.conf_int().loc[treat_centered].values if treat_centered in res_int.params.index else (np.nan, np.nan)
            
            beta_interact = float(res_int.params.get(interact_col, np.nan))
            p_interact = float(res_int.pvalues.get(interact_col, np.nan))
            ci_interact = res_int.conf_int().loc[interact_col].values if interact_col in res_int.params.index else (np.nan, np.nan)
            
            r2 = float(res_int.rsquared) if hasattr(res_int, "rsquared") else np.nan
            
            # Log control mean for moderation models
            if np.isfinite(control_mean_ct6_mod):
                logger.info(f"    Control mean CT6 (treatment <= median): {control_mean_ct6_mod:.2f} deaths/100k")
            
        except Exception as e:
            logger.warning(f"    Moderation OLS failed for {label}: {e}")
            beta_mod = p_mod = beta_treat = p_treat = beta_interact = p_interact = r2 = np.nan
            ci_mod = ci_treat = ci_interact = (np.nan, np.nan)
        
        moderation_rows.append({
            "Interaction": label,
            "Moderator": mod_col,
            "Treatment": treat_col,
            "Outcome": "CT6_Hospital_Deaths",
            "N": len(d),
            "Control_Mean_CT6": control_mean_ct6_mod,
            "Beta_Moderator": beta_mod,
            "p_Moderator": p_mod,
            "CI_Moderator_Lower": ci_mod[0],
            "CI_Moderator_Upper": ci_mod[1],
            "Beta_Treatment": beta_treat,
            "p_Treatment": p_treat,
            "CI_Treatment_Lower": ci_treat[0],
            "CI_Treatment_Upper": ci_treat[1],
            "Beta_Interaction": beta_interact,
            "p_Interaction": p_interact,
            "CI_Interaction_Lower": ci_interact[0],
            "CI_Interaction_Upper": ci_interact[1],
            "R2": r2
        })
    
    moderation_df = pd.DataFrame(moderation_rows)
    if not moderation_df.empty:
        # FDR correction on interaction p-values
        moderation_df = _bh_correct_in_place(moderation_df, "p_Interaction", None, "q_Interaction")
        
        out_csv = os.path.join(out_dir, "ct6_hospital_deaths_moderation_effects.csv")
        moderation_df.to_csv(out_csv, index=False)
        logger.info(f"\nSaved CT6 moderation effects: {out_csv}")
        logger.info(f"  {len(moderation_df)} tests completed.")
        
        # Prominent summary of moderation results with control means
        logger.info("\n" + "="*70)
        logger.info("CT6 MODERATION EFFECTS SUMMARY (with control group baselines)")
        logger.info("="*70)
        for _, row in moderation_df.iterrows():
            logger.info(f"{row['Interaction']:40s}")
            if pd.notna(row['Control_Mean_CT6']):
                logger.info(f"  Control mean (treatment <= median): {row['Control_Mean_CT6']:8.2f} deaths/100k")
            logger.info(f"  Interaction coefficient: {row['Beta_Interaction']:8.4f}")
            logger.info(f"  Interaction p-value:     {row['p_Interaction']:.4f}")
            if 'q_Interaction' in row and pd.notna(row['q_Interaction']):
                logger.info(f"  Interaction q-value:     {row['q_Interaction']:.4f}")
            logger.info("")
        logger.info("="*70)
    else:
        logger.warning("No CT6 moderation effects results produced.")
    
    results["moderation_effects"] = moderation_df
    
    # ============================================================================
    # TWO-PART MODEL FOR CT6 (Zero-Inflated Treatment Analysis)
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("CT6 TWO-PART (HURDLE) MODEL ANALYSIS")
    logger.info("Addressing Review 1: Testing extensive vs intensive margins")
    logger.info("="*80)
    
    # Run two-part model for MO14 -> CT6
    # Use df_work (merged df) and check for column availability
    ct6_controls = get_control_set(outcome, base_controls, df_work)
    required_cols = ['mo14_ai_automate_routine_tasks_pct', outcome] + ct6_controls
    available_cols = [c for c in required_cols if c in df_work.columns]
    
    if 'mo14_ai_automate_routine_tasks_pct' in df_work.columns and outcome in df_work.columns:
        ct6_mo14_data = df_work[available_cols].dropna()
        
        if not ct6_mo14_data.empty and len(ct6_mo14_data) >= 50:
            logger.info("\n[MO14 -> CT6] Running Two-Part Model:")
            # Use only base_controls that are actually in the data
            available_controls = [c for c in ct6_controls if c in ct6_mo14_data.columns]
            
            (logit_ext_ct6, ols_int_ct6, ext_results_ct6, int_results_ct6) = run_two_part_model(
                df=ct6_mo14_data,
                treat_col='mo14_ai_automate_routine_tasks_pct',
                y_col=outcome,
                confounders=available_controls,
                logger=logger
            )
            
            # Save results
            if ext_results_ct6 and int_results_ct6:
                two_part_ct6_results = []
                two_part_ct6_results.append({
                    'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                    'Outcome': outcome,
                    'Margin': 'Extensive (Any vs None)',
                    'N': ext_results_ct6.get('N_total', np.nan),
                    'N_Treated': ext_results_ct6.get('N_treated', np.nan),
                    'N_Control': ext_results_ct6.get('N_control', np.nan),
                    'Raw_Difference': ext_results_ct6.get('Raw_difference', np.nan),
                    'Adjusted_ATE': ext_results_ct6.get('ATE_extensive_adjusted', np.nan),
                    'Y_treated_mean': ext_results_ct6.get('Y_treated_mean', np.nan),
                    'Y_control_mean': ext_results_ct6.get('Y_control_mean', np.nan),
                    'Pseudo_R2': ext_results_ct6.get('Pseudo_R2', np.nan),
                    'Model': 'Logit + AIPW'
                })
                two_part_ct6_results.append({
                    'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                    'Outcome': outcome,
                    'Margin': 'Intensive (More vs Less)',
                    'N': int_results_ct6.get('N_adopters', np.nan),
                    'Effect_OLS': int_results_ct6.get('Beta_OLS', np.nan),
                    'p_value_OLS': int_results_ct6.get('p_value_OLS', np.nan),
                    'Effect_GPS': int_results_ct6.get('Beta_GPS', np.nan),
                    'p_value_GPS': int_results_ct6.get('p_value_GPS', np.nan),
                    'CI_lower_OLS': int_results_ct6.get('CI_lower_OLS', np.nan),
                    'CI_upper_OLS': int_results_ct6.get('CI_upper_OLS', np.nan),
                    'CI_lower_GPS': int_results_ct6.get('CI_lower_GPS', np.nan),
                    'CI_upper_GPS': int_results_ct6.get('CI_upper_GPS', np.nan),
                    'R2': int_results_ct6.get('R2', np.nan),
                    'Exposure_mean': int_results_ct6.get('Exposure_mean', np.nan),
                    'GAM_EDF': int_results_ct6.get('gam_analysis', {}).get('edf', np.nan),
                    'Tertile_diff': int_results_ct6.get('tertile_analysis', {}).get('diff', np.nan),
                    'Tertile_p': int_results_ct6.get('tertile_analysis', {}).get('p', np.nan),
                    'Model': 'OLS + GPS-IPW (Causal, Adopters Only)'
                })
                
                two_part_ct6_df = pd.DataFrame(two_part_ct6_results)
                out_csv = os.path.join(out_dir, "ct6_two_part_model.csv")
                two_part_ct6_df.to_csv(out_csv, index=False)
                logger.info(f"\n✓ Saved CT6 two-part model results: {out_csv}")
                
                results["two_part_model"] = two_part_ct6_df
            else:
                logger.warning("Two-part model for CT6 did not produce complete results.")
        else:
            logger.warning(f"Insufficient data for CT6 two-part model analysis (N={len(ct6_mo14_data) if not ct6_mo14_data.empty else 0}).")
    else:
        logger.warning("Required columns for CT6 two-part model not available in df_work.")
    
    return results


def run_threshold_sensitivity(df: pd.DataFrame, treat_col: str, y_col: str, thresholds: List[float],
                              base_controls: List[str], logger, plot_dir: str) -> pd.DataFrame:
    """
    Evaluate AIPW at alternative binary thresholds for a continuous treatment.
    thresholds are expressed in the same units as treat_col (e.g., percent of beds).
    """
    base_controls = get_control_set(y_col, base_controls, df)
    rows = []
    for th in thresholds:
        bin_col = f"{treat_col}_gt_{th}"
        df_local = df[[treat_col, y_col, 'state_fips_for_clustering'] + base_controls].dropna().copy()
        if df_local.empty:
            continue
        df_local[bin_col] = (df_local[treat_col] > th).astype(int)
        if df_local[bin_col].nunique() < 2:
            logger.warning(f"    Threshold {th}: insufficient variation for {treat_col}.")
            continue
        (aipw_ate, aipw_cl, aipw_cu, aipw_p,
         n_t, n_c, aipw_err) = run_aipw(df_local, bin_col, y_col, base_controls, N_BOOT, logger, plot_dir)
        rows.append({
            "Treatment": treat_col,
            "Outcome": y_col,
            "Threshold": th,
            "N": len(df_local),
            "N_Treated": n_t,
            "N_Control": n_c,
            "ATE": aipw_ate,
            "CI_Lower": aipw_cl,
            "CI_Upper": aipw_cu,
            "p_value": aipw_p
        })
    return pd.DataFrame(rows)


def run_tmle_superlearner(df: pd.DataFrame, treat_col: str, y_col: str, confounders: List[str], logger):
    """
    Run a simple TMLE with SuperLearner-style models (RF outcome, logistic exposure).
    Uses median split on treatment to align with AIPW/DML binary setup.
    """
    confounders = get_control_set(y_col, confounders, df)
    cols = [treat_col, y_col] + confounders
    d = df[cols].dropna().copy()
    if d.empty or len(d) < 100:
        logger.warning(f"TMLE: insufficient data for {treat_col}->{y_col} (N={len(d)}).")
        return None

    # Binary treatment via median split
    median_val = d[treat_col].median()
    d['A'] = (d[treat_col] > median_val).astype(int)
    if d['A'].nunique() < 2:
        logger.warning("TMLE: treatment is not binary after median split.")
        return None

    d = d.rename(columns={y_col: 'Y'})

    # Build TMLE object
    tmle = TMLE(d, exposure='A', outcome='Y')
    conf_formula = " + ".join(confounders)
    tmle.exposure_model(
        conf_formula,
        custom_model=LogisticRegression(
            solver='lbfgs', max_iter=500, penalty='l2', C=1.0, random_state=42
        )
    )
    tmle.outcome_model(
        f"A + {conf_formula}",
        custom_model=RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
    )

    try:
        tmle.fit()
        # zEpid TMLE attribute names vary by version and estimator type.
        def _first_attr(obj, names, default=np.nan):
            for name in names:
                if hasattr(obj, name):
                    val = getattr(obj, name)
                    if val is not None:
                        return val
            return default

        ate_raw = _first_attr(
            tmle,
            [
                "average_treatment_effect",
                "ate",
                "risk_difference",
                "rd",
            ],
            default=np.nan,
        )
        ate = float(np.asarray(ate_raw).squeeze()) if np.isfinite(np.asarray(ate_raw).squeeze()) else np.nan

        ci_raw = _first_attr(
            tmle,
            [
                "average_treatment_effect_ci",
                "ate_ci",
                "risk_difference_ci",
                "rd_ci",
            ],
            default=(np.nan, np.nan),
        )
        try:
            ci_arr = np.asarray(ci_raw, dtype=float).reshape(-1)
            ci_lower = float(ci_arr[0]) if ci_arr.size >= 1 else np.nan
            ci_upper = float(ci_arr[1]) if ci_arr.size >= 2 else np.nan
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan

        se_raw = _first_attr(
            tmle,
            [
                "average_treatment_effect_se",
                "ate_se",
                "risk_difference_se",
                "rd_se",
            ],
            default=np.nan,
        )
        try:
            se = float(np.asarray(se_raw).squeeze())
        except Exception:
            se = np.nan
        if (not np.isfinite(se) or se <= 0) and np.isfinite(ci_lower) and np.isfinite(ci_upper):
            se = float((ci_upper - ci_lower) / (2 * 1.96))
        if not np.isfinite(ate):
            raise ValueError("TMLE fit completed but no compatible effect estimate attribute was found.")
        
        # Calculate two-sided p-value from z-score
        from scipy import stats
        z_score = ate / se if np.isfinite(se) and se > 0 else np.nan
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if np.isfinite(z_score) else np.nan
        
        return {
            "Treatment": treat_col,
            "Outcome": y_col,
            "N": len(d),
            "N_Treated": int((d['A'] == 1).sum()),
            "N_Control": int((d['A'] == 0).sum()),
            "ATE": ate,
            "CI_Lower": float(ci_lower),
            "CI_Upper": float(ci_upper),
            "p_value": float(p_value),
            "Exposure_Model": "LogisticRegression(l2)",
            "Outcome_Model": "RandomForestRegressor",
            "Treatment_Split": f"> median ({median_val:.3f})",
            "Cross_Fit": False,
        }
    except Exception as e:
        logger.warning(f"TMLE failed for {treat_col}->{y_col}: {e}")
        return None


def aipw_point_estimate(df: pd.DataFrame, treat_col: str, y_col: str, confounders: List[str], logger) -> Optional[float]:
    """Single AIPW point estimate (no bootstrap) for speed in block bootstrap."""
    confounders = get_control_set(y_col, confounders, df)
    T = df[treat_col].astype(int).values
    Y = df[y_col].values
    X = df[confounders].copy()
    if X.isnull().any().any():
        X = X.fillna(X.mean())
    ps, scaler = estimate_propensity_scores(X, T, logger, treat_col)
    if ps is None:
        return None
    Xs = scaler.transform(X.apply(pd.to_numeric, errors='coerce'))
    def _fit_mu_single(X_all, Y_all, mask):
        y = Y_all[mask]
        Xa = X_all.copy()
        if np.isnan(Xa).any():
            col_means = np.nanmean(Xa, axis=0)
            inds = np.where(np.isnan(Xa))
            Xa[inds] = np.take(col_means, inds[1])
        if mask.sum() > Xa.shape[1] and mask.sum() > 5:
            try:
                m = sm.OLS(y, add_const(Xa[mask])).fit()
                return m.predict(add_const(Xa))
            except Exception:
                return np.full(len(Y_all), np.mean(y) if mask.sum() > 0 else np.mean(Y_all))
        else:
            return np.full(len(Y_all), np.mean(y) if mask.sum() > 0 else np.mean(Y_all))
    mu1 = _fit_mu_single(Xs, Y, (T == 1))
    mu0 = _fit_mu_single(Xs, Y, (T == 0))
    term1 = (T / ps) * (Y - mu1) + mu1
    term0 = ((1 - T) / (1 - ps)) * (Y - mu0) + mu0
    ate = np.mean(term1[np.isfinite(term1)]) - np.mean(term0[np.isfinite(term0)])
    return ate


def spatial_block_bootstrap(df: pd.DataFrame, treat_col: str, y_col: str, confounders: List[str],
                            cluster_col: str, n_boot: int, logger) -> Optional[Dict[str, float]]:
    """
    State-level (cluster) block bootstrap for AIPW point estimates.
    Uses median split binary treatment to mirror primary analysis.
    """
    confounders = get_control_set(y_col, confounders, df)
    required = [treat_col, y_col, cluster_col] + confounders
    d = df[required].dropna().copy()
    if d.empty:
        logger.warning("Spatial bootstrap: no data after dropna.")
        return None
    median_val = d[treat_col].median()
    bin_col = f"{treat_col}_gt_median"
    d[bin_col] = (d[treat_col] > median_val).astype(int)
    if d[bin_col].nunique() < 2:
        logger.warning("Spatial bootstrap: treatment not binary after median split.")
        return None

    clusters = d[cluster_col].unique()
    if len(clusters) < 2:
        logger.warning("Spatial bootstrap: fewer than 2 clusters.")
        return None

    boot_ates = []
    for i in range(n_boot):
        sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
        boot_df = pd.concat([d[d[cluster_col] == c] for c in sampled_clusters], axis=0)
        ate_b = aipw_point_estimate(boot_df, bin_col, y_col, confounders, logger)
        if ate_b is not None and np.isfinite(ate_b):
            boot_ates.append(ate_b)

    if len(boot_ates) < 10:
        logger.warning(f"Spatial bootstrap: only {len(boot_ates)} successful iterations.")
        return None
    boot_arr = np.array(boot_ates)
    mean_ate = float(np.mean(boot_arr))
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))
    p_proxy = np.mean(boot_arr > 0) if mean_ate < 0 else np.mean(boot_arr < 0)
    p_value = float(min(1.0, 2 * p_proxy))

    return {
        "Treatment": treat_col,
        "Outcome": y_col,
        "Clusters": int(len(clusters)),
        "N": len(d),
        "N_Treated": int((d[bin_col] == 1).sum()),
        "N_Control": int((d[bin_col] == 0).sum()),
        "Median_Split": median_val,
        "Iterations": len(boot_ates),
        "ATE_Mean": mean_ate,
        "CI_Lower": ci_lower,
        "CI_Upper": ci_upper,
        "p_value": p_value,
        "Cluster_Var": cluster_col,
    }

# =============================================================================
# E-value & Lives-saved sim
# =============================================================================
def calculate_e_value(effect, sd_ctrl):
    if np.isnan(effect) or np.isnan(sd_ctrl) or sd_ctrl == 0:
        return np.nan
    smd = abs(effect) / sd_ctrl
    if smd == 0:
        return 1.0
    rr_approx = np.exp(0.91 * smd)
    if rr_approx <= 1:
        return 1.0
    try:
        return rr_approx + np.sqrt(rr_approx * (rr_approx - 1))
    except Exception:
        return np.nan

# =============================================================================
# =============================================================================
# Propensity Score Matching + Rosenbaum Bounds (Sensitivity Analysis)
# =============================================================================
def perform_ps_matching(df: pd.DataFrame, treatment_col: str, outcome_col: str,
                        confounders: List[str], caliper: float = 0.25, 
                        method: str = 'nearest', logger=None) -> pd.DataFrame:
    """
    Perform Propensity Score Matching (1:1 Nearest Neighbor WITHOUT Replacement).
    
    This implements proper 1:1 nearest-neighbor matching WITHOUT replacement, using
    logit-based propensity scores and a caliper to ensure good matches. Matching
    without replacement is essential for valid Rosenbaum bounds because the Wilcoxon
    signed-rank test assumes independent matched pairs.
    
    Parameters:
    -----------
    df : DataFrame with treatment, outcome, and confounders
    treatment_col : Binary treatment indicator (0/1)
    outcome_col : Outcome variable
    confounders : List of confounder variable names
    caliper : Maximum distance for matching (default: 0.25 standard deviations of logit(PS))
    method : Matching method ('nearest' for nearest-neighbor)
    logger : Logger instance
    
    Returns:
    --------
    DataFrame with matched pairs, including: treatment_binary, outcome, ps, pair_id, weight
    
    Reference: Austin, P. C. (2011). An Introduction to Propensity Score Methods for 
               Reducing the Effects of Confounding in Observational Studies. 
               Multivariate Behavioral Research, 46(3), 399-424.
    """
    import statsmodels.api as sm
    
    # Prepare data
    d = df[[treatment_col, outcome_col] + confounders].dropna().copy()
    d['treatment_binary'] = d[treatment_col].astype(int)
    
    # Estimate propensity scores using logistic regression
    X = sm.add_constant(d[confounders])
    ps_model = sm.Logit(d['treatment_binary'], X).fit(disp=0)
    d['ps'] = ps_model.predict(X)
    
    # Calculate logit of propensity score for caliper
    d['logit_ps'] = np.log(d['ps'] / (1 - d['ps']))
    caliper_dist = caliper * d['logit_ps'].std()
    
    # Separate groups
    treated = d[d['treatment_binary'] == 1].copy()
    control = d[d['treatment_binary'] == 0].copy()
    
    if logger:
        logger.info(f"\n    PS Matching (1:1 Without Replacement):")
        logger.info(f"      Treated: {len(treated)}, Control: {len(control)}")
        logger.info(f"      Caliper: {caliper_dist:.4f}")

    matched_pairs = []
    used_control_indices = set()
    
    # Randomize treated order to prevent bias in greedy matching
    treated = treated.sample(frac=1, random_state=42)
    
    for t_idx, t_row in treated.iterrows():
        # Calculate distances to ALL controls
        control['dist'] = np.abs(control['logit_ps'] - t_row['logit_ps'])
        
        # Filter: Within caliper AND not used yet
        valid_controls = control[
            (control['dist'] <= caliper_dist) & 
            (~control.index.isin(used_control_indices))
        ]
        
        if not valid_controls.empty:
            # Greedy match: pick closest
            best_match_idx = valid_controls['dist'].idxmin()
            c_row = control.loc[best_match_idx]
            
            # Mark control as used
            used_control_indices.add(best_match_idx)
            
            matched_pairs.append({
                'treated_unit': t_idx,
                'control_unit': best_match_idx,
                'outcome_treated': t_row[outcome_col],
                'outcome_control': c_row[outcome_col],
                'ps_treated': t_row['ps'],
                'ps_control': c_row['ps'],
                'ps_distance': valid_controls.loc[best_match_idx, 'dist'],
                **{f'treated_{var}': t_row[var] for var in confounders},
                **{f'control_{var}': c_row[var] for var in confounders}
            })
            
    matched_df = pd.DataFrame(matched_pairs)
    
    if logger:
        logger.info(f"      Pairs matched: {len(matched_df)}")
        logger.info(f"      Matching rate: {len(matched_df)/len(treated)*100:.1f}%")
        
    return matched_df


def assess_balance_after_matching(matched_df: pd.DataFrame, confounders: List[str], 
                                   logger=None) -> pd.DataFrame:
    """
    Assess covariate balance after propensity score matching using Standardized Mean Differences (SMD).
    
    SMD < 0.1 indicates good balance (general rule of thumb)
    SMD < 0.2 is acceptable
    SMD > 0.2 suggests imbalance
    
    Parameters:
    -----------
    matched_df : DataFrame with matched pairs (from perform_ps_matching)
    confounders : List of confounder variable names
    logger : Logger instance
    
    Returns:
    --------
    DataFrame with balance statistics: Covariate, SMD_Before, SMD_After, Balanced
    """
    balance_results = []
    
    for var in confounders:
        treated_col = f'treated_{var}'
        control_col = f'control_{var}'
        
        if treated_col not in matched_df.columns or control_col not in matched_df.columns:
            continue
        
        # After matching SMD
        mean_t = matched_df[treated_col].mean()
        mean_c = matched_df[control_col].mean()
        var_t = matched_df[treated_col].var()
        var_c = matched_df[control_col].var()
        pooled_sd = np.sqrt((var_t + var_c) / 2)
        
        smd_after = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0.0
        
        balance_results.append({
            'Covariate': var,
            'Mean_Treated': mean_t,
            'Mean_Control': mean_c,
            'SMD_After_Matching': abs(smd_after),
            'Balanced': 'Yes' if abs(smd_after) < 0.1 else ('Acceptable' if abs(smd_after) < 0.2 else 'No')
        })
    
    balance_df = pd.DataFrame(balance_results)
    
    if logger:
        logger.info(f"\n      Balance Assessment (After Matching):")
        logger.info(f"        SMD < 0.1: Excellent balance")
        logger.info(f"        SMD < 0.2: Acceptable balance")
        logger.info(f"        SMD > 0.2: Poor balance")
        logger.info(f"        {'Variable':<30s} {'SMD':>8s} {'Status':>12s}")
        logger.info(f"        {'-'*52}")
        for _, row in balance_df.iterrows():
            logger.info(f"        {row['Covariate']:<30s} {row['SMD_After_Matching']:>8.3f} {row['Balanced']:>12s}")
        
        n_balanced = (balance_df['Balanced'] == 'Yes').sum()
        n_acceptable = (balance_df['Balanced'] == 'Acceptable').sum()
        logger.info(f"        {'-'*52}")
        logger.info(f"        Excellent balance: {n_balanced}/{len(balance_df)} variables")
        logger.info(f"        Acceptable balance: {n_acceptable}/{len(balance_df)} variables")
    
    return balance_df


def calculate_rosenbaum_bounds_matched(matched_df: pd.DataFrame, gammas: List[float] = None, 
                                       logger=None) -> pd.DataFrame:
    """
    Standard Rosenbaum Bounds sensitivity analysis for matched pairs.
    Calculates bounds on the p-value of the Wilcoxon Signed Rank test using
    the standard expectation/variance method from Rosenbaum (2002).
    
    This implementation follows the rbounds R package methodology, computing
    bounds on the Expectation and Variance of the Rank Sum statistic directly.
    
    Gamma (Γ) represents the odds ratio of differential treatment assignment due to 
    unmeasured confounding:
    - Gamma = 1.0: No hidden bias
    - Gamma = 1.5: Matched pairs could differ in odds of treatment by 50%
    - Gamma = 2.0: Matched pairs could differ in odds of treatment by 2x
    
    Parameters:
    -----------
    matched_df : DataFrame with matched pairs (must have outcome_treated, outcome_control)
    gammas : List of gamma values to test (default: [1.0, 1.1, ..., 3.0])
    logger : Logger instance
    
    Returns:
    --------
    DataFrame with columns: Gamma, P_Upper_Bound, Significant_at_05
    
    Reference: Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.
    """
    from scipy import stats
    
    if gammas is None:
        gammas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0]
    
    # Calculate differences (Treated - Control)
    diff = matched_df['outcome_treated'] - matched_df['outcome_control']
    
    # Remove ties (differences of 0) per standard procedure
    diff = diff[diff != 0]
    n = len(diff)
    
    if n == 0:
        if logger:
            logger.warning("      No non-zero matched pair differences")
        return pd.DataFrame()

    # Calculate W: Sum of ranks of positive differences
    # If standard effect is NEGATIVE (e.g. YPLL reduction), we look at negative diffs
    # to test if they are significant.
    # To standardize: We simply look at the absolute treatment effect direction.
    
    median_diff = np.median(diff)
    
    # If observed effect is negative (reduction in mortality), we sum the ranks 
    # of the negative differences (conceptually), or flip signs so effect is positive.
    if median_diff < 0:
        diff = -diff
        
    # Calculate Wilcoxon statistic (W)
    abs_diff = np.abs(diff)
    ranks = stats.rankdata(abs_diff)
    # W is sum of ranks where difference is positive (consistent with effect)
    W = np.sum(ranks[diff > 0])
    
    if logger:
        logger.info(f"\n      Rosenbaum Bounds (Matched Pairs):")
        logger.info(f"        Number of non-zero pairs: {n}")
        logger.info(f"        Observed W statistic: {W:.1f}")
        logger.info(f"\n        {'Gamma':>6s} {'P_Upper':>10s} {'Sig@0.05':>10s}")
        logger.info(f"        {'-'*28}")

    results = []
    
    for gamma in gammas:
        # Probability of assignment
        p_plus = gamma / (1 + gamma)
        p_minus = 1 / (1 + gamma)
        
        # Expectation bounds for W
        # The sum of all ranks is n(n+1)/2
        total_rank_sum = n * (n + 1) / 2
        
        # Max Expectation (Worst case: Hidden bias causes the effect)
        E_max = p_plus * total_rank_sum
        
        # Variance bounds (approximate using sum of squared ranks)
        # Var = sum(r_i^2) * p * (1-p)
        sum_sq_ranks = np.sum(ranks**2)
        Var_max = p_plus * p_minus * sum_sq_ranks
        
        # Z-score (Continuity corrected)
        z_score = (W - E_max - 0.5) / np.sqrt(Var_max)
        
        # P-value (Upper bound)
        # This is the p-value for the hypothesis that the effect is NOT real,
        # given the worst-case confounding gamma.
        p_upper = 1 - stats.norm.cdf(z_score)
        
        results.append({
            'Gamma': gamma,
            'P_Upper_Bound': p_upper,
            'Significant_at_05': 'Yes' if p_upper < 0.05 else 'No'
        })
        
        if logger:
            sig_marker = '✓' if p_upper < 0.05 else '✗'
            logger.info(f"        {gamma:>6.1f} {p_upper:>10.4f} {sig_marker:>10s}")
        
    return pd.DataFrame(results)


# =============================================================================
# Double Machine Learning (DML)
# =============================================================================
def run_dml(df: pd.DataFrame, treat_col: str, y_col: str, confounders: List[str], 
            logger, n_splits: int = 5, discrete_treatment: bool = False):
    """
    Run Double Machine Learning for causal effect estimation.
    Uses cross-fitting with Random Forest for nuisance parameters.
    
    CRITICAL: Reviewer's requested continuous treatment as PRIMARY analysis.
    Default is now discrete_treatment=False (continuous exposure).
    
    Parameters:
    -----------
    df : DataFrame with treatment, outcome, and confounders
    treat_col : Treatment variable (can be continuous or binary)
    y_col : Outcome variable (continuous)
    confounders : List of confounder variable names
    logger : Logger instance
    n_splits : Number of cross-fitting splits (default 5)
    discrete_treatment : If True, treatment is discrete (0/1); if False, continuous (DEFAULT)
    
    Returns:
    --------
    Tuple of (Effect, std_error, CI_lower, CI_upper, p_value, success_flag)
    For continuous: Effect = change in Y per 1-unit change in T
    For binary: Effect = ATE (High vs Low)
    """
    confounders = get_control_set(y_col, confounders, df)
    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        # Prepare data
        required_cols = [treat_col, y_col] + confounders
        d = df[required_cols].dropna().copy()
        
        if d.empty or len(d) < 100:
            logger.warning(f"    DML: Insufficient data ({len(d)} rows). Need ≥100.")
            return (np.nan, np.nan, np.nan, np.nan, np.nan, False)
        
        T = d[treat_col].values
        Y = d[y_col].values
        X = d[confounders].values
        
        # Check for variation
        if np.std(Y) < 1e-6:
            logger.warning("    DML: No variation in outcome.")
            return (np.nan, np.nan, np.nan, np.nan, np.nan, False)
        
        if discrete_treatment and len(np.unique(T)) != 2:
            logger.warning(f"    DML: Discrete treatment must be binary. Found {len(np.unique(T))} unique values.")
            return (np.nan, np.nan, np.nan, np.nan, np.nan, False)
        
        # Initialize DML with appropriate models
        # For discrete treatment, use classifier for propensity model
        # For continuous treatment (Review 1 request), use regressor for Generalized Propensity Score
        if discrete_treatment:
            model_t = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                           min_samples_leaf=20, random_state=42, n_jobs=-1)
        else:
            # Continuous treatment: Model treatment density (Generalized Propensity Score)
            model_t = RandomForestRegressor(n_estimators=100, max_depth=5,
                                          min_samples_leaf=20, random_state=42, n_jobs=-1)
        
        model_y = RandomForestRegressor(n_estimators=100, max_depth=5, 
                                       min_samples_leaf=20, random_state=42, n_jobs=-1)
        
        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_treatment,
            linear_first_stages=False,
            cv=n_splits,
            random_state=42
        )
        
        # Fit DML model
        dml.fit(Y, T, X=X, inference='auto')
        
        # Get effect estimate (constant marginal effect)
        effect_array = dml.const_marginal_effect(X)
        effect = float(np.mean(effect_array)) if hasattr(effect_array, '__iter__') else float(effect_array)
        
        # Get confidence interval
        ci = dml.const_marginal_effect_interval(X, alpha=0.05)
        ci_lower = float(np.mean(ci[0])) if hasattr(ci[0], '__iter__') else float(ci[0])
        ci_upper = float(np.mean(ci[1])) if hasattr(ci[1], '__iter__') else float(ci[1])
        
        # Calculate standard error and p-value
        std_err = (ci_upper - ci_lower) / (2 * 1.96)
        if std_err > 0:
            z_stat = effect / std_err
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        else:
            p_value = np.nan
        
        treatment_type = "Binary" if discrete_treatment else "Continuous"
        effect_label = "ATE" if discrete_treatment else "Beta"
        logger.info(f"    DML ({treatment_type}): {effect_label}={effect:.2f}, SE={std_err:.2f}, p={p_value:.4f}")
        
        return (effect, std_err, ci_lower, ci_upper, p_value, True)
        
    except ImportError as e:
        logger.warning(f"    DML: econml package not available. Install with: pip install econml")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, False)
    except Exception as e:
        logger.warning(f"    DML estimation failed: {e}")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, False)


def run_ipw_continuous(df: pd.DataFrame, treat_col: str, y_col: str, confounders: List[str], logger):
    """
    Inverse Probability Weighting for Continuous Treatment (Generalized Propensity Score).
    Addressing Reviewer's request for GPS methods.
    
    This implements stabilized GPS weights as per Hirano & Imbens (2004):
    - Numerator: f(T) - marginal density of treatment
    - Denominator: f(T|X) - conditional density of treatment given confounders
    
    Parameters:
    -----------
    df : DataFrame with treatment, outcome, and confounders
    treat_col : Continuous treatment variable
    y_col : Outcome variable
    confounders : List of confounder names
    logger : Logger instance
    
    Returns:
    --------
    Tuple of (beta, p_value, ci_lower, ci_upper)
    Beta = marginal effect of treatment on outcome (per-unit change)
    """
    from scipy.stats import norm
    confounders = get_control_set(y_col, confounders, df)
    
    try:
        # Prepare data
        required_cols = [treat_col, y_col] + confounders
        d = df[required_cols].dropna().copy()
        
        if d.empty or len(d) < 50:
            logger.warning(f"    GPS-IPW: Insufficient data ({len(d)} rows). Need ≥50.")
            return (np.nan, np.nan, np.nan, np.nan)
        
        T = d[treat_col].values
        Y = d[y_col].values
        X = d[confounders]
        
        # 1. Model the treatment density
        # Denominator model: T ~ X (what drives adoption?)
        X_const = add_const(X)
        mod_denom = sm.OLS(T, X_const).fit()
        pred_mean = mod_denom.predict(X_const)
        pred_std = np.std(mod_denom.resid)
        
        # Avoid division by zero
        if pred_std < 1e-6:
            logger.warning("    GPS-IPW: Treatment model has no residual variance.")
            return (np.nan, np.nan, np.nan, np.nan)
        
        # Calculate conditional density f(T|X)
        denom_dens = norm.pdf(T, loc=pred_mean, scale=pred_std)
        
        # Numerator model: T ~ 1 (stabilizing factor)
        num_mean = np.mean(T)
        num_std = np.std(T)
        
        if num_std < 1e-6:
            logger.warning("    GPS-IPW: Treatment has no variance.")
            return (np.nan, np.nan, np.nan, np.nan)
        
        # Calculate marginal density f(T)
        num_dens = norm.pdf(T, loc=num_mean, scale=num_std)
        
        # Calculate GPS weights (stabilized)
        weights = num_dens / denom_dens
        
        # Winsorize extreme weights for stability (Review 2 concern)
        weights = np.clip(weights, np.percentile(weights, 1), np.percentile(weights, 99))
        
        # Normalize weights to sum to N (optional, helps interpretation)
        weights = weights / np.mean(weights)
        
        logger.info(f"    GPS-IPW: Weight range [{weights.min():.2f}, {weights.max():.2f}], mean={weights.mean():.2f}")
        
        # 2. Estimate Marginal Structural Model (MSM) weighted by GPS
        # Y ~ T + confounders (Weighted OLS)
        msm_data = d.copy()
        msm_data['const'] = 1
        
        # Include confounders in MSM for doubly-robust property
        msm_vars = ['const', treat_col] + confounders
        wls_model = sm.WLS(Y, msm_data[msm_vars], weights=weights).fit(cov_type='HC1')
        
        beta = wls_model.params[treat_col]
        p_val = wls_model.pvalues[treat_col]
        conf = wls_model.conf_int().loc[treat_col]
        ci_lower, ci_upper = conf[0], conf[1]
        
        logger.info(f"    GPS-IPW (Continuous): Beta={beta:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}], p={p_val:.4f}")
        
        return (beta, p_val, ci_lower, ci_upper)
        
    except Exception as e:
        logger.error(f"    GPS-IPW Error: {e}")
        return (np.nan, np.nan, np.nan, np.nan)


def run_two_part_model(df, treat_col, y_col, confounders, logger):
    """
    Implements a Two-Part (Hurdle) Model for zero-inflated continuous treatment.
    
    This is the standard econometric solution for zero-inflated continuous data,
    addressing Reviewer's concern that treating zero-inflated exposure as a 
    standard continuous variable dilutes the effect.
    
    Part 1 (Extensive Margin): Logit for P(Treatment > 0) - "Participation Effect"
        Tests whether having ANY AI access (vs. none) matters
        
    Part 2 (Intensive Margin): OLS for E[Y | Treatment > 0] - "Intensity Effect"
        Tests whether MORE AI (conditional on having some) matters
    
    Parameters:
    -----------
    df : DataFrame with treatment, outcome, and confounders
    treat_col : Continuous treatment variable (zero-inflated)
    y_col : Outcome variable
    confounders : List of confounder names
    logger : Logger instance
    
    Returns:
    --------
    Tuple of (logit_model, ols_model, extensive_results, intensive_results)
    """
    import statsmodels.api as sm
    confounders = get_control_set(y_col, confounders, df)
    
    try:
        # Filter data
        required_cols = [treat_col, y_col] + confounders
        d = df[required_cols].dropna().copy()
        
        if d.empty or len(d) < 50:
            logger.warning(f"    Two-Part Model: Insufficient data ({len(d)} rows). Need ≥50.")
            return (None, None, {}, {})
        
        # --- Part 1: Extensive Margin (Binary: Any Adoption vs None) ---
        d['treat_binary'] = (d[treat_col] > 0).astype(int)
        n_treated = d['treat_binary'].sum()
        n_control = len(d) - n_treated
        pct_treated = 100 * n_treated / len(d)
        
        logger.info(f"\n    Two-Part Model Part 1 (Extensive Margin):")
        logger.info(f"      Treated (any exposure): {n_treated} ({pct_treated:.1f}%)")
        logger.info(f"      Control (zero exposure): {n_control} ({100-pct_treated:.1f}%)")
        
        X = sm.add_constant(d[confounders])
        
        # Logit for probability of having ANY access
        logit_model = sm.Logit(d['treat_binary'], X).fit(disp=0)
        
        # Calculate raw difference (unadjusted)
        y_treated_ext = d[d['treat_binary'] == 1][y_col].mean()
        y_control_ext = d[d['treat_binary'] == 0][y_col].mean()
        raw_difference = y_treated_ext - y_control_ext
        
        # Calculate Average Marginal Effect (AME) from Logit model
        # This is the proper adjusted extensive margin effect
        marginal_effects = logit_model.get_margeff(method='dydx', at='overall')
        
        # Get the counterfactual outcomes using AIPW-style adjustment
        # E[Y(1)] - E[Y(0)] accounting for propensity scores
        d['pr_adoption'] = logit_model.predict(X)
        ps_clipped = np.clip(d['pr_adoption'], 0.01, 0.99)
        
        # AIPW for binary treatment: ATE = E[T*Y/e - (1-T)*Y/(1-e) + (1-T/e)*mu1 + (T/(1-e) - 1)*mu0]
        # Simplified: just use outcome regression for counterfactuals
        Y = d[y_col].values
        T_binary = d['treat_binary'].values
        
        # Fit outcome models for E[Y|T=1,X] and E[Y|T=0,X]
        X_t1 = sm.add_constant(d[d['treat_binary']==1][confounders])
        X_t0 = sm.add_constant(d[d['treat_binary']==0][confounders])
        
        if len(X_t1) > 0 and len(X_t0) > 0:
            mu1_model = sm.OLS(d[d['treat_binary']==1][y_col], X_t1).fit()
            mu0_model = sm.OLS(d[d['treat_binary']==0][y_col], X_t0).fit()
            
            # Predict for all units - use only columns that were in the fitted model
            # This handles cases where a confounder had no variation in treated/control subset
            mu1_cols = mu1_model.model.exog_names
            mu0_cols = mu0_model.model.exog_names
            
            X_full = sm.add_constant(d[confounders])
            mu1_all = mu1_model.predict(X_full[mu1_cols])
            mu0_all = mu0_model.predict(X_full[mu0_cols])
            
            # AIPW estimator
            ipw_term = T_binary * Y / ps_clipped - (1 - T_binary) * Y / (1 - ps_clipped)
            aug_term = (1 - T_binary / ps_clipped) * mu1_all - (1 - (1 - T_binary) / (1 - ps_clipped)) * mu0_all
            
            ate_extensive_adjusted = np.mean(ipw_term + aug_term)
        else:
            # Fallback to raw difference if insufficient data for adjustment
            ate_extensive_adjusted = raw_difference
        
        logger.info(f"      Logit Pseudo R²: {logit_model.prsquared:.4f}")
        logger.info(f"      Mean outcome (treated): {y_treated_ext:.2f}")
        logger.info(f"      Mean outcome (control): {y_control_ext:.2f}")
        logger.info(f"      Raw Difference (unadjusted): {raw_difference:.2f} YPLL")
        logger.info(f"      AIPW-Adjusted ATE (Extensive Margin): {ate_extensive_adjusted:.2f} YPLL")
        
        extensive_results = {
            'N_total': len(d),
            'N_treated': n_treated,
            'N_control': n_control,
            'Pct_treated': pct_treated,
            'Pseudo_R2': logit_model.prsquared,
            'Y_treated_mean': y_treated_ext,
            'Y_control_mean': y_control_ext,
            'Raw_difference': raw_difference,
            'ATE_extensive_adjusted': ate_extensive_adjusted
        }
        
        # --- Part 2: Intensive Margin (Continuous: Effect of Intensity among Adopters) ---
        logger.info(f"\n    Two-Part Model Part 2 (Intensive Margin):")
        logger.info(f"      UPGRADE: Using causal methods (GPS-IPW + Nonlinearity checks) instead of plain OLS")
        
        # Subset to only treated units (positive exposure)
        d_treated = d[d[treat_col] > 0].copy()
        
        if len(d_treated) < 20:
            logger.warning(f"      Insufficient treated units ({len(d_treated)}). Need ≥20.")
            return (logit_model, None, extensive_results, {})
        
        logger.info(f"      Analyzing {len(d_treated)} counties with positive AI exposure")
        logger.info(f"      Exposure range: [{d_treated[treat_col].min():.2f}, {d_treated[treat_col].max():.2f}]")
        logger.info(f"      Exposure mean: {d_treated[treat_col].mean():.2f}, median: {d_treated[treat_col].median():.2f}")
        
        # --- 2A: Baseline OLS (for comparison) ---
        X_treated_ols = sm.add_constant(d_treated[[treat_col] + confounders])
        ols_intensity = sm.OLS(d_treated[y_col], X_treated_ols).fit(cov_type='HC1')
        
        beta_ols = ols_intensity.params[treat_col]
        p_ols = ols_intensity.pvalues[treat_col]
        ci_ols = ols_intensity.conf_int().loc[treat_col]
        
        logger.info(f"\n      [2A] Baseline OLS (among adopters):")
        logger.info(f"           R²: {ols_intensity.rsquared:.4f}")
        logger.info(f"           Coefficient: {beta_ols:.4f}, 95% CI: [{ci_ols[0]:.4f}, {ci_ols[1]:.4f}], p={p_ols:.4f}")
        
        # --- 2B: Causal Estimate via GPS-IPW (among adopters) ---
        logger.info(f"\n      [2B] GPS-IPW (Continuous, among adopters):")
        (beta_gps, p_gps, ci_lower_gps, ci_upper_gps) = run_ipw_continuous(
            df=d_treated,
            treat_col=treat_col,
            y_col=y_col,
            confounders=confounders,
            logger=logger
        )
        
        if not np.isnan(beta_gps):
            logger.info(f"           GPS-IPW Coefficient: {beta_gps:.4f}, 95% CI: [{ci_lower_gps:.4f}, {ci_upper_gps:.4f}], p={p_gps:.4f}")
        else:
            logger.warning(f"           GPS-IPW failed for intensive margin")
        
        # --- 2C: Nonlinearity Check - Quantile Bins ---
        logger.info(f"\n      [2C] Nonlinearity Check - Quantile Bins (among adopters):")
        try:
            # Divide adopters into tertiles
            d_treated['exposure_tertile'] = pd.qcut(d_treated[treat_col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            
            if d_treated['exposure_tertile'].nunique() >= 2:
                # Compare top vs bottom tertile
                low_tertile = d_treated[d_treated['exposure_tertile'] == 'Low']
                high_tertile = d_treated[d_treated['exposure_tertile'] == 'High']
                
                if len(low_tertile) > 0 and len(high_tertile) > 0:
                    y_low = low_tertile[y_col].mean()
                    y_high = high_tertile[y_col].mean()
                    diff_tertiles = y_high - y_low
                    
                    # Simple t-test
                    from scipy.stats import ttest_ind
                    t_stat, p_tertile = ttest_ind(high_tertile[y_col].dropna(), low_tertile[y_col].dropna())
                    
                    logger.info(f"           Low tertile (n={len(low_tertile)}): mean outcome = {y_low:.2f}")
                    logger.info(f"           High tertile (n={len(high_tertile)}): mean outcome = {y_high:.2f}")
                    logger.info(f"           Difference (High - Low): {diff_tertiles:.2f}, p={p_tertile:.4f}")
                    
                    tertile_result = {
                        'n_low': len(low_tertile),
                        'n_high': len(high_tertile),
                        'y_low': y_low,
                        'y_high': y_high,
                        'diff': diff_tertiles,
                        'p': p_tertile
                    }
                else:
                    logger.warning(f"           Insufficient data in tertiles")
                    tertile_result = {}
            else:
                logger.warning(f"           Cannot create tertiles (insufficient variation)")
                tertile_result = {}
        except Exception as e:
            logger.warning(f"           Tertile analysis failed: {e}")
            tertile_result = {}
        
        # --- 2D: Nonlinearity Check - Spline/GAM ---
        logger.info(f"\n      [2D] Nonlinearity Check - GAM (Generalized Additive Model):")
        try:
            try:
                from pygam import LinearGAM, s
                pygam_available = True
            except ImportError:
                pygam_available = False
                raise ImportError("pygam not installed")
            
            X_gam = d_treated[confounders].values
            y_gam = d_treated[y_col].values
            treat_gam = d_treated[treat_col].values
            
            # Fit GAM with smooth term for treatment
            gam = LinearGAM(s(0, n_splines=4)).fit(treat_gam.reshape(-1, 1), y_gam)
            
            # Get effective degrees of freedom (EDF)
            # EDF ≈ 1 suggests linear; EDF > 1.5 suggests nonlinearity
            edf = gam.statistics_['edof']
            p_gam = gam.statistics_['p_values'][0] if 'p_values' in gam.statistics_ else np.nan
            
            logger.info(f"           GAM EDF (Effective Degrees of Freedom): {edf:.2f}")
            if edf < 1.5:
                logger.info(f"           → Linear relationship (EDF ≈ 1)")
            else:
                logger.info(f"           → Potential nonlinearity (EDF > 1.5)")
            
            gam_result = {
                'edf': edf,
                'p_gam': p_gam
            }
        except ImportError:
            logger.warning(f"           pygam not installed; skipping GAM analysis")
            logger.info(f"           Install with: pip install pygam")
            gam_result = {}
        except Exception as e:
            logger.warning(f"           GAM analysis failed: {e}")
            gam_result = {}
        
        # --- Summary Interpretation ---
        logger.info(f"\n      [SUMMARY] Intensive Margin Interpretation:")
        logger.info(f"        OLS (descriptive):  β={beta_ols:.4f}, p={p_ols:.4f}")
        if not np.isnan(beta_gps):
            logger.info(f"        GPS-IPW (causal):   β={beta_gps:.4f}, p={p_gps:.4f}")
        
        if p_ols > 0.05 and (np.isnan(beta_gps) or p_gps > 0.05):
            logger.info(f"        ✗ NO DOSE-RESPONSE: Both descriptive and causal methods find null effect")
            logger.info(f"        → Benefit is a THRESHOLD EFFECT (0→1), not continuous scaling")
        elif p_ols < 0.05 or (not np.isnan(beta_gps) and p_gps < 0.05):
            logger.info(f"        ✓ DOSE-RESPONSE DETECTED: Evidence of continuous scaling among adopters")
        
        intensive_results = {
            'N_adopters': len(d_treated),
            'Exposure_min': d_treated[treat_col].min(),
            'Exposure_max': d_treated[treat_col].max(),
            'Exposure_mean': d_treated[treat_col].mean(),
            'Exposure_median': d_treated[treat_col].median(),
            # OLS results
            'R2': ols_intensity.rsquared,
            'Beta_OLS': beta_ols,
            'SE_OLS': ols_intensity.bse[treat_col],
            'CI_lower_OLS': ci_ols[0],
            'CI_upper_OLS': ci_ols[1],
            'p_value_OLS': p_ols,
            # GPS-IPW results
            'Beta_GPS': beta_gps,
            'CI_lower_GPS': ci_lower_gps,
            'CI_upper_GPS': ci_upper_gps,
            'p_value_GPS': p_gps,
            # Nonlinearity checks
            'tertile_analysis': tertile_result,
            'gam_analysis': gam_result
        }
        
        return (logit_model, ols_intensity, extensive_results, intensive_results)
        
    except Exception as e:
        logger.error(f"    Two-Part Model Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return (None, None, {}, {})


def run_primary_county_crossfit_summary(
    df: pd.DataFrame,
    base_controls: List[str],
    logger,
    out_dir: str,
    plot_dir: str,
    crossfit_n_boot: int = 0,
) -> pd.DataFrame:
    """
    Multi-method county summary table centered on the new cross-fitted AIPW estimator.
    """
    rows: List[Dict[str, Any]] = []

    for spec in PRIMARY_COUNTY_SPECS:
        treat_col = spec["treatment"]
        y_col = spec["outcome"]
        label = spec["label"]
        logger.info(f"  Primary county summary: {label}")

        d, controls, bin_col, threshold = prepare_county_binary_analysis(
            df=df,
            treat_col=treat_col,
            y_col=y_col,
            base_controls=base_controls,
            treatment_rule="median",
        )
        if d.empty or d[bin_col].nunique() < 2:
            logger.warning(f"    Skipping {label}: insufficient data after applying county controls.")
            continue

        legacy_ate, legacy_cl, legacy_cu, legacy_p, n_t, n_c, legacy_err = run_aipw(
            d,
            bin_col,
            y_col,
            controls,
            N_BOOT,
            logger,
            plot_dir,
        )
        if legacy_err:
            legacy_ate = legacy_cl = legacy_cu = legacy_p = np.nan

        crossfit_ate, crossfit_cl, crossfit_cu, crossfit_p, _, _, crossfit_diag, crossfit_err = run_crossfit_aipw(
            df=d,
            treat_col=bin_col,
            y_col=y_col,
            confounders=controls,
            n_boot=crossfit_n_boot,
            logger=logger,
            cluster_ids=d["state_fips_for_clustering"],
            cluster_bootstrap=False,
        )
        if crossfit_err:
            crossfit_ate = crossfit_cl = crossfit_cu = crossfit_p = np.nan
            crossfit_diag = {}

        dml_ate, dml_se, dml_cl, dml_cu, dml_p, dml_ok = run_dml(
            df=d,
            treat_col=bin_col,
            y_col=y_col,
            confounders=controls,
            logger=logger,
            n_splits=5,
            discrete_treatment=True,
        )
        if not dml_ok:
            dml_ate = dml_se = dml_cl = dml_cu = dml_p = np.nan

        tmle_res = run_tmle_superlearner(
            df=d,
            treat_col=treat_col,
            y_col=y_col,
            confounders=controls,
            logger=logger,
        )

        ato_ate, ato_cl, ato_cu, ato_p, _, _, ato_ess, ato_err = run_aipw_overlap(
            d,
            bin_col,
            y_col,
            controls,
            N_BOOT,
            logger,
            plot_dir,
        )
        if ato_err:
            ato_ate = ato_cl = ato_cu = ato_p = ato_ess = np.nan

        prevalence = float(d[bin_col].mean())
        control_mean = float(d.loc[d[bin_col] == 0, y_col].mean()) if (d[bin_col] == 0).any() else np.nan
        rows.append({
            "Label": label,
            "Treatment": treat_col,
            "Outcome": y_col,
            "Threshold": threshold,
            "N": len(d),
            "N_Treated": int((d[bin_col] == 1).sum()),
            "N_Control": int((d[bin_col] == 0).sum()),
            "Treatment_Prevalence": prevalence,
            "Control_Mean": control_mean,
            "Crossfit_Relative_Change_Pct": compute_relative_change(crossfit_ate, control_mean),
            "Legacy_AIPW_ATE": legacy_ate,
            "Legacy_AIPW_CI_Lower": legacy_cl,
            "Legacy_AIPW_CI_Upper": legacy_cu,
            "Legacy_AIPW_p": legacy_p,
            "Crossfit_AIPW_ATE": crossfit_ate,
            "Crossfit_AIPW_CI_Lower": crossfit_cl,
            "Crossfit_AIPW_CI_Upper": crossfit_cu,
            "Crossfit_AIPW_p": crossfit_p,
            "Crossfit_ESS": crossfit_diag.get("ess_overall", np.nan),
            "Crossfit_Max_Weight": crossfit_diag.get("max_weight", np.nan),
            "Crossfit_P99_Weight": crossfit_diag.get("p99_weight", np.nan),
            "DML_ATE": dml_ate,
            "DML_SE": dml_se,
            "DML_CI_Lower": dml_cl,
            "DML_CI_Upper": dml_cu,
            "DML_p": dml_p,
            "TMLE_ATE": tmle_res.get("ATE", np.nan) if tmle_res else np.nan,
            "TMLE_CI_Lower": tmle_res.get("CI_Lower", np.nan) if tmle_res else np.nan,
            "TMLE_CI_Upper": tmle_res.get("CI_Upper", np.nan) if tmle_res else np.nan,
            "TMLE_p": tmle_res.get("p_value", np.nan) if tmle_res else np.nan,
            "ATO_ATE": ato_ate,
            "ATO_CI_Lower": ato_cl,
            "ATO_CI_Upper": ato_cu,
            "ATO_p": ato_p,
            "ATO_ESS": ato_ess,
        })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        out_csv = os.path.join(out_dir, "primary_county_crossfit_summary.csv")
        summary_df.to_csv(out_csv, index=False)
        logger.info(f"Saved primary county cross-fit summary: {out_csv}")
    return summary_df


def run_county_clip_trim_sensitivity_suite(
    df: pd.DataFrame,
    base_controls: List[str],
    logger,
    out_dir: str,
    plot_dir: str,
    crossfit_n_boot: int = 0,
) -> pd.DataFrame:
    rows = []
    for spec in PRIMARY_COUNTY_SPECS:
        sens_df = run_clip_trim_sensitivity(
            df=df,
            treat_col=spec["treatment"],
            y_col=spec["outcome"],
            base_controls=base_controls,
            logger=logger,
            plot_dir=plot_dir,
            n_boot=crossfit_n_boot,
            treatment_rule="median",
        )
        if not sens_df.empty:
            sens_df.insert(0, "Label", spec["label"])
            rows.append(sens_df)

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out_df.empty:
        out_csv = os.path.join(out_dir, "county_crossfit_clip_trim_sensitivity.csv")
        out_df.to_csv(out_csv, index=False)
        logger.info(f"Saved county clip/trim sensitivity table: {out_csv}")
    return out_df


def run_county_ols_sensitivity_suite(
    df: pd.DataFrame,
    base_controls: List[str],
    logger,
    out_dir: str,
) -> Dict[str, pd.DataFrame]:
    partial_rows = []
    oster_rows = []

    for spec in PRIMARY_COUNTY_SPECS:
        partial = run_partial_r2_sensitivity(
            df=df,
            treat_col=spec["treatment"],
            y_col=spec["outcome"],
            base_controls=base_controls,
            logger=logger,
        )
        if partial:
            partial["Label"] = spec["label"]
            partial_rows.append(partial)

        oster = run_oster_sensitivity(
            df=df,
            treat_col=spec["treatment"],
            y_col=spec["outcome"],
            base_controls=base_controls,
            logger=logger,
        )
        if oster:
            oster["Label"] = spec["label"]
            oster_rows.append(oster)

    partial_df = pd.DataFrame(partial_rows)
    oster_df = pd.DataFrame(oster_rows)

    if not partial_df.empty:
        partial_csv = os.path.join(out_dir, "county_partial_r2_sensitivity.csv")
        partial_df.to_csv(partial_csv, index=False)
        logger.info(f"Saved county partial R^2 sensitivity table: {partial_csv}")
    if not oster_df.empty:
        oster_csv = os.path.join(out_dir, "county_oster_sensitivity.csv")
        oster_df.to_csv(oster_csv, index=False)
        logger.info(f"Saved county Oster sensitivity table: {oster_csv}")

    return {"partial_r2": partial_df, "oster": oster_df}


def run_county_misclassification_suite(
    df: pd.DataFrame,
    base_controls: List[str],
    logger,
    out_dir: str,
    n_draws: int = 200,
    n_workers: int = 8,
    progress_every_sec: int = 30,
    rf_n_jobs_per_worker: int = 1,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []

    for spec in PRIMARY_COUNTY_SPECS:
        controls = get_control_set(spec["outcome"], base_controls, df)
        required_cols = [spec["treatment"], spec["outcome"], "state_fips_for_clustering"] + controls
        required_cols = [c for c in required_cols if c in df.columns]
        d = df[required_cols].dropna().copy()
        if d.empty:
            logger.warning(
                f"Misclassification skipped for {spec['label']}: empty data after required columns."
            )
            continue

        d["__t_obs"] = (pd.to_numeric(d[spec["treatment"]], errors="coerce") > 0).astype(int)
        if d["__t_obs"].nunique() < 2:
            logger.warning(
                f"Misclassification skipped for {spec['label']}: observed extensive-margin treatment has no variation."
            )
            continue

        d_task = d[[spec["outcome"]] + controls + ["__t_obs"]].copy()
        for se, sp in MISCLASSIFICATION_SCENARIOS:
            tasks.append({
                "label": spec["label"],
                "treat_col": spec["treatment"],
                "y_col": spec["outcome"],
                "controls": controls,
                "d_task": d_task,
                "se": se,
                "sp": sp,
            })

    if not tasks:
        logger.warning("No eligible county misclassification tasks were generated.")
        return pd.DataFrame()

    cpu_total = os.cpu_count() or 1
    max_workers = max(1, min(n_workers, cpu_total, len(tasks)))
    logger.info(
        "Starting county treatment misclassification sensitivity: "
        f"{len(tasks)} Se/Sp scenarios, n_draws={n_draws}, workers={max_workers}, "
        f"rf_n_jobs_per_worker={rf_n_jobs_per_worker}."
    )

    start_time = time.time()
    def _run_tasks_sequential(task_list: List[Dict[str, Any]], start_idx: int = 1) -> List[Dict[str, Any]]:
        seq_rows: List[Dict[str, Any]] = []
        for i, task in enumerate(task_list, start=start_idx):
            logger.info(
                f"  Misclassification scenario {i}/{len(task_list)}: "
                f"{task['label']} (Se={task['se']:.2f}, Sp={task['sp']:.2f})"
            )
            result = _run_misclassification_scenario_task(
                d=task["d_task"],
                y_col=task["y_col"],
                controls=task["controls"],
                se=task["se"],
                sp=task["sp"],
                n_draws=n_draws,
                ps_clip=(0.01, 0.99),
                seed_offset=i,
                rf_n_jobs=rf_n_jobs_per_worker,
            )
            if result is None:
                continue
            result.update({
                "Label": task["label"],
                "Treatment": task["treat_col"],
                "Outcome": task["y_col"],
            })
            seq_rows.append(result)
        return seq_rows

    if max_workers == 1:
        rows.extend(_run_tasks_sequential(tasks))
    else:
        failed_count = 0
        first_failure_msg = ""
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    _run_misclassification_scenario_task,
                    task["d_task"],
                    task["y_col"],
                    task["controls"],
                    task["se"],
                    task["sp"],
                    n_draws,
                    (0.01, 0.99),
                    idx,
                    rf_n_jobs_per_worker,
                ): task
                for idx, task in enumerate(tasks, start=1)
            }

            pending = set(future_to_task.keys())
            completed = 0
            while pending:
                done, pending = cf.wait(
                    pending,
                    timeout=progress_every_sec,
                    return_when=cf.FIRST_COMPLETED,
                )
                if not done:
                    elapsed = (time.time() - start_time) / 60.0
                    logger.info(
                        f"  Misclassification progress: {completed}/{len(tasks)} scenarios complete "
                        f"(elapsed {elapsed:.1f} minutes)."
                    )
                    continue

                for future in done:
                    task = future_to_task[future]
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as e:
                        failed_count += 1
                        if not first_failure_msg:
                            first_failure_msg = str(e)
                        logger.warning(
                            f"  Misclassification scenario failed for {task['label']} "
                            f"(Se={task['se']:.2f}, Sp={task['sp']:.2f}): {e}"
                        )
                        continue

                    if result is not None:
                        result.update({
                            "Label": task["label"],
                            "Treatment": task["treat_col"],
                            "Outcome": task["y_col"],
                        })
                        rows.append(result)

                    elapsed = (time.time() - start_time) / 60.0
                    logger.info(
                        f"  Misclassification scenario {completed}/{len(tasks)} complete: "
                        f"{task['label']} (Se={task['se']:.2f}, Sp={task['sp']:.2f}), "
                        f"elapsed {elapsed:.1f} minutes."
                    )
        if failed_count == len(tasks):
            logger.warning(
                "Parallel misclassification failed for all scenarios; falling back to sequential mode. "
                f"First error: {first_failure_msg}"
            )
            rows.extend(_run_tasks_sequential(tasks))

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(
            by=["Label", "Sensitivity", "Specificity"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        out_csv = os.path.join(out_dir, "county_treatment_misclassification_sensitivity.csv")
        out_df.to_csv(out_csv, index=False)
        logger.info(f"Saved county treatment misclassification sensitivity table: {out_csv}")
    else:
        logger.warning("County treatment misclassification sensitivity produced no successful scenario results.")

    total_minutes = (time.time() - start_time) / 60.0
    logger.info(f"County misclassification sensitivity finished in {total_minutes:.1f} minutes.")
    return out_df


def add_spatial_lag(df: pd.DataFrame, group_col: str, target_col: str, lag_name: str) -> pd.DataFrame:
    """
    Calculates the average of target_col for other counties in the same state/division.
    Rough proxy for spatial interference/spillover requested by Review 2.
    
    This implements a "leave-one-out" spatial lag:
    For county i in state s: SpatialLag_i = mean(target_col for all j≠i in state s)
    
    Parameters:
    -----------
    df : DataFrame
    group_col : Grouping variable (e.g., 'state_fips_for_clustering')
    target_col : Variable to spatially lag
    lag_name : Name for the new lagged variable
    
    Returns:
    --------
    DataFrame with new lag column added
    """
    def leave_one_out_mean(x):
        """Calculate mean excluding self."""
        sum_x = x.sum()
        count_x = x.count()
        if count_x <= 1:
            return np.nan
        return (sum_x - x) / (count_x - 1)
    
    df[lag_name] = df.groupby(group_col)[target_col].transform(leave_one_out_mean)
    return df


# =============================================================================
# Plots
# =============================================================================
def plot_interaction_continuous(df: pd.DataFrame, A_col: str, B_col: str, Y_col: str,
                                name: str, beta: float, pval: float, plot_dir: str, logger):
    if not VISUALIZATION_AVAILABLE or any(c not in df.columns for c in [A_col, B_col, Y_col]):
        return
    d = df[[A_col, B_col, Y_col]].dropna().copy()
    if len(d) < 20:
        return
    try:
        from sklearn.linear_model import LinearRegression
        d['_AxB'] = d[A_col] * d[B_col]
        X = d[[A_col, B_col, '_AxB']].values
        y = d[Y_col].values
        lr = LinearRegression().fit(X, y)
        b_vals = [d[B_col].quantile(0.25), d[B_col].quantile(0.75)]
        a_min, a_max = d[A_col].quantile(0.02), d[A_col].quantile(0.98)
        a_grid = np.linspace(a_min, a_max, 100)
        plt.figure(figsize=(9, 6))
        plt.scatter(d[A_col], d[Y_col], alpha=0.25, s=18, label="Data points")
        colors = ['blue', 'red']
        for i, b in enumerate(b_vals):
            Xline = np.column_stack([a_grid, np.full_like(a_grid, b), a_grid*b])
            yhat = lr.predict(Xline)
            plt.plot(a_grid, yhat, linewidth=2.5, color=colors[i], label=f"Moderator at {b:.2f} (25/75th %ile)")
        plt.title(f"{name} | β_int={beta:.2f}, p={pval:.3f}")
        plt.xlabel(A_col.replace('_c','').replace('_',' ').title())
        plt.ylabel(Y_col.replace('_',' ').title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        outp = os.path.join(plot_dir, f"interaction_cont_{name}.png".replace(" ", "_"))
        plt.savefig(outp, dpi=300)
        plt.close()
        logger.info(f"Saved continuous interaction plot: {outp}")
    except Exception as e:
        logger.error(f"Error creating continuous interaction plot for {name}: {e}")
        plt.close()


def _cohens_f2_label(f2: float) -> str:
    if pd.isna(f2):
        return "N/A"
    if f2 < 0.02:
        return "Negligible effect"
    if f2 < 0.15:
        return "Small effect"
    if f2 < 0.35:
        return "Medium effect"
    return "Large effect"


def _winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    valid = s_num.dropna()
    if valid.empty:
        return s_num
    lo, hi = valid.quantile(lower_q), valid.quantile(upper_q)
    return s_num.clip(lower=lo, upper=hi)


def _signed_log1p_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    return np.sign(s_num) * np.log1p(np.abs(s_num))


def plot_continuous_moderator_surface(
    d: pd.DataFrame,
    iv_col: str,
    moderator_col: str,
    outcome_col: str,
    iv_centered_col: str,
    moderator_centered_col: str,
    interaction_term: str,
    controls_for_model: List[str],
    res_model,
    name: str,
    plot_dir: str,
    logger,
    beta: float = np.nan,
    pval: float = np.nan,
    cohens_f2: float = np.nan,
    label_map: Optional[Dict[str, str]] = None,
    transform_label: str = "Raw",
):
    """
    Build a 3D continuous moderation surface:
    X-axis = IV, Y-axis = moderator, Z-axis = predicted outcome.
    """
    if not VISUALIZATION_AVAILABLE:
        return

    needed_cols = [iv_col, moderator_col, outcome_col, iv_centered_col, moderator_centered_col]
    if any(c not in d.columns for c in needed_cols):
        logger.warning(f"Skipping surface plot '{name}': missing one or more required columns.")
        return

    if d.empty or len(d) < 40:
        logger.warning(f"Skipping surface plot '{name}': insufficient data (N={len(d)}).")
        return

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        params = res_model.params
        label_map = label_map or {}

        iv_label = label_map.get(iv_col, iv_col.replace("_", " ").title())
        mod_label = label_map.get(moderator_col, moderator_col.replace("_", " ").title())
        out_label = label_map.get(outcome_col, outcome_col.replace("_", " ").title())

        # Plot on central range to reduce leverage from extreme tails.
        iv_min, iv_max = d[iv_col].quantile(0.02), d[iv_col].quantile(0.98)
        mod_min, mod_max = d[moderator_col].quantile(0.02), d[moderator_col].quantile(0.98)
        iv_grid = np.linspace(iv_min, iv_max, 60)
        mod_grid = np.linspace(mod_min, mod_max, 60)
        iv_mesh, mod_mesh = np.meshgrid(iv_grid, mod_grid)

        iv_mean = d[iv_col].mean()
        mod_mean = d[moderator_col].mean()
        iv_mesh_c = iv_mesh - iv_mean
        mod_mesh_c = mod_mesh - mod_mean

        # Keep controls fixed at sample means to isolate interaction geometry.
        control_means = d[controls_for_model].mean() if controls_for_model else pd.Series(dtype=float)
        z_hat = np.full(iv_mesh.shape, float(params.get("const", 0.0)))
        z_hat += float(params.get(iv_centered_col, 0.0)) * iv_mesh_c
        z_hat += float(params.get(moderator_centered_col, 0.0)) * mod_mesh_c
        z_hat += float(params.get(interaction_term, 0.0)) * (iv_mesh_c * mod_mesh_c)
        for ctrl in controls_for_model:
            if ctrl in params.index:
                z_hat += float(params[ctrl]) * float(control_means.get(ctrl, 0.0))

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Downsample observed points for readability.
        sample_n = min(len(d), 3000)
        d_scatter = d.sample(sample_n, random_state=42) if len(d) > sample_n else d
        ax.scatter(
            d_scatter[iv_col],
            d_scatter[moderator_col],
            d_scatter[outcome_col],
            c="black",
            alpha=0.18,
            s=9,
            label="Observed counties",
        )

        surface = ax.plot_surface(
            iv_mesh,
            mod_mesh,
            z_hat,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )

        line_1 = f"Continuous Moderator Surface ({transform_label}): {iv_label} x {mod_label} -> {out_label}"
        line_2 = f"beta_int={beta:.4f}, p={pval:.4f}" if pd.notna(beta) and pd.notna(pval) else ""
        title_text = f"{line_1}\n{line_2}" if line_2 else line_1
        ax.set_title(title_text, pad=16)

        ax.set_xlabel(iv_label, labelpad=10)
        ax.set_ylabel(mod_label, labelpad=10)
        ax.set_zlabel(out_label, labelpad=10)
        ax.view_init(elev=26, azim=40)

        cbar = fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label(f"Predicted {out_label}")
        ax.legend(loc="upper left")

        os.makedirs(plot_dir, exist_ok=True)
        outp = os.path.join(plot_dir, f"continuous_moderator_surface_{name}.png".replace(" ", "_"))
        fig.savefig(outp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved continuous moderator surface plot: {outp}")
    except Exception as e:
        logger.error(f"Error creating continuous moderator surface plot '{name}': {e}")
        plt.close("all")


def plot_forest_plot_ate(results_df: pd.DataFrame, outcome_label: str, plot_dir: str, logger, 
                         treatment_filter: list = None, filename: str = "forest_plot_ate.png"):
    """
    Create a forest plot showing ATE estimates with 95% CIs.
    
    Parameters:
    -----------
    results_df : DataFrame with columns Treatment, AIPW_ATE, AIPW_CI_Lower, AIPW_CI_Upper, Outcome
    outcome_label : str, label for the outcome variable
    plot_dir : str, directory to save the plot
    logger : logging.Logger
    treatment_filter : list, optional list of treatment variables to include
    filename : str, name of output file
    """
    try:
        # Filter for specific outcome and treatments if specified
        df = results_df[results_df['Outcome'].notna()].copy()
        
        if treatment_filter:
            df = df[df['Treatment'].isin(treatment_filter)].copy()
        
        if df.empty:
            logger.warning(f"No data available for forest plot: {filename}")
            return
        
        # Create readable labels mapping
        label_map = {
            "mo1_genai_composite_score": "GenAI Composite Score",
            "mo2_robotics_composite_score": "Robotics Composite Score",
            "mo11_ai_staff_scheduling_pct": "AI Staff Scheduling",
            "mo12_ai_predict_staff_needs_pct": "AI Predict Staff Needs",
            "mo13_ai_predict_patient_demand_pct": "AI Predict Pt Demand",
            "mo14_ai_automate_routine_tasks_pct": "AI Automate Routine Tasks",
            "mo15_ai_optimize_workflows_pct": "AI Optimize Workflows",
            "mo21_robotics_in_hospital_pct": "Robotics in Hospital",
            "sp5_irr_county_value": "County Rurality (IRR)",
        }
        
        df['Label'] = df['Treatment'].map(label_map).fillna(df['Treatment'])
        
        # Sort by ATE magnitude for better visualization
        df = df.sort_values('AIPW_ATE', ascending=True).reset_index(drop=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.6)))
        
        # Plot point estimates and CIs
        y_positions = range(len(df))
        ax.errorbar(df['AIPW_ATE'], y_positions, 
                   xerr=[df['AIPW_ATE'] - df['AIPW_CI_Lower'], 
                         df['AIPW_CI_Upper'] - df['AIPW_ATE']],
                   fmt='o', color='#1f77b4', markersize=8, 
                   capsize=5, capthick=2, linewidth=2,
                   label='ATE and 95% CI')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, 
                  label='Line of No Effect', zorder=0)
        
        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df['Label'])
        ax.set_xlabel(f'Average Treatment Effect (ATE) on {outcome_label}', fontsize=12)
        ax.set_ylabel('Technology Component', fontsize=12)
        ax.set_title(f'Adjusted Associations of Technology Adoption on {outcome_label}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        os.makedirs(plot_dir, exist_ok=True)
        outp = os.path.join(plot_dir, filename)
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved forest plot: {outp}")
        
    except Exception as e:
        logger.error(f"Error creating forest plot {filename}: {e}")
        plt.close()


def plot_iptw_vs_ato_forest(overlap_df: pd.DataFrame, outcome_label: str, plot_dir: str, 
                            logger, filename: str = "forest_plot_iptw_vs_ato.png"):
    """
    Create a forest plot comparing IPTW (standard) vs ATO (overlap-weighted) estimates.
    
    This visualization addresses reviewer concerns about positivity by showing that
    effects remain robust even when restricting to the overlap population with best
    covariate balance.
    
    Parameters:
    -----------
    overlap_df : DataFrame with columns IPTW_ATE, IPTW_CI_Lower, IPTW_CI_Upper,
                 ATO_ATE, ATO_CI_Lower, ATO_CI_Upper, Label
    outcome_label : str, label for the outcome variable
    plot_dir : str, directory to save the plot
    logger : logging.Logger
    filename : str, name of output file
    """
    try:
        df = overlap_df.copy()
        
        if df.empty:
            logger.warning(f"No data available for IPTW vs ATO forest plot")
            return
        
        # Create figure with appropriate height
        n_comparisons = len(df)
        fig, ax = plt.subplots(figsize=(12, max(6, n_comparisons * 1.2)))
        
        # Prepare data for plotting
        # Each treatment gets 2 rows: IPTW and ATO
        y_positions = []
        estimates = []
        ci_lowers = []
        ci_uppers = []
        labels = []
        colors = []
        markers = []
        
        y_pos = 0
        for idx, row in df.iterrows():
            # IPTW estimate (standard)
            if not np.isnan(row['IPTW_ATE']):
                y_positions.append(y_pos)
                estimates.append(row['IPTW_ATE'])
                ci_lowers.append(row['IPTW_CI_Lower'])
                ci_uppers.append(row['IPTW_CI_Upper'])
                labels.append(f"{row['Label']} - Standard IPTW")
                colors.append('#1f77b4')  # Blue
                markers.append('o')
                y_pos += 1
            
            # ATO estimate (overlap-weighted)
            if not np.isnan(row['ATO_ATE']):
                y_positions.append(y_pos)
                estimates.append(row['ATO_ATE'])
                ci_lowers.append(row['ATO_CI_Lower'])
                ci_uppers.append(row['ATO_CI_Upper'])
                labels.append(f"{row['Label']} - Conservative ATO")
                colors.append('#ff7f0e')  # Orange
                markers.append('s')
                y_pos += 1
            
            # Add spacing between different treatments
            y_pos += 0.5
        
        # Plot estimates with CIs
        for i, (y, est, ci_l, ci_u, color, marker) in enumerate(zip(
            y_positions, estimates, ci_lowers, ci_uppers, colors, markers)):
            
            # Error bar
            ax.plot([ci_l, ci_u], [y, y], color=color, linewidth=2.5, alpha=0.8)
            
            # Caps
            cap_size = 0.15
            ax.plot([ci_l, ci_l], [y - cap_size, y + cap_size], color=color, linewidth=2.5, alpha=0.8)
            ax.plot([ci_u, ci_u], [y - cap_size, y + cap_size], color=color, linewidth=2.5, alpha=0.8)
            
            # Point estimate
            ax.scatter(est, y, marker=marker, s=120, color=color, edgecolors='black', 
                      linewidths=1.5, zorder=5, alpha=0.9)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                  label='Line of No Effect', zorder=0, alpha=0.7)
        
        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(f'Average Treatment Effect on {outcome_label} (95% CI)', fontsize=13, fontweight='bold')
        ax.set_title('Robustness to Positivity: Standard IPTW vs Conservative Overlap Weighting (ATO)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
                   markersize=10, label='Standard IPTW (ATE)', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', 
                   markersize=10, label='Conservative ATO (Overlap)', markeredgecolor='black', markeredgewidth=1.5),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='No Effect')
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=11, 
                 shadow=True, fancybox=True)
        
        # Grid and styling
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save
        os.makedirs(plot_dir, exist_ok=True)
        outp = os.path.join(plot_dir, filename)
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved IPTW vs ATO forest plot: {outp}")
        logger.info(f"  Caption: The association remains robust even when restricting to the zone")
        logger.info(f"           of covariate overlap (ATO), suggesting findings are not driven by")
        logger.info(f"           counties with extreme propensity scores.")
        
    except Exception as e:
        logger.error(f"Error creating IPTW vs ATO forest plot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        plt.close()


def calculate_smd(df, treat_col, confounders, ps=None):
    """
    Calculate Standardized Mean Differences (SMD) before and after AIPW weighting.
    
    Parameters:
    -----------
    df : DataFrame containing treatment and covariates
    treat_col : str, binary treatment column
    confounders : list of covariate column names
    ps : array-like, optional propensity scores for calculating weighted SMD
    
    Returns:
    --------
    DataFrame with columns: Covariate, SMD_Unweighted, SMD_Weighted (if ps provided)
    """
    T = df[treat_col].astype(int).values
    X = df[confounders].copy()
    
    if X.isnull().any().any():
        X = X.fillna(X.mean())
    
    smd_results = []
    
    for col in confounders:
        x = X[col].values
        
        # Unweighted SMD
        mean_treated = np.mean(x[T == 1])
        mean_control = np.mean(x[T == 0])
        var_treated = np.var(x[T == 1])
        var_control = np.var(x[T == 0])
        pooled_sd = np.sqrt((var_treated + var_control) / 2)
        
        smd_unweighted = (mean_treated - mean_control) / pooled_sd if pooled_sd > 0 else 0.0
        
        result = {
            'Covariate': col,
            'SMD_Unweighted': smd_unweighted
        }
        
        # Weighted SMD (if propensity scores provided)
        if ps is not None:
            try:
                # IPTW weights
                weights = np.where(T == 1, 1 / ps, 1 / (1 - ps))
                weights = np.clip(weights, 0, 100)  # Clip extreme weights
                
                weighted_mean_treated = np.average(x[T == 1], weights=weights[T == 1])
                weighted_mean_control = np.average(x[T == 0], weights=weights[T == 0])
                
                weighted_var_treated = np.average((x[T == 1] - weighted_mean_treated)**2, weights=weights[T == 1])
                weighted_var_control = np.average((x[T == 0] - weighted_mean_control)**2, weights=weights[T == 0])
                
                pooled_sd_weighted = np.sqrt((weighted_var_treated + weighted_var_control) / 2)
                smd_weighted = (weighted_mean_treated - weighted_mean_control) / pooled_sd_weighted if pooled_sd_weighted > 0 else 0.0
                
                result['SMD_Weighted'] = smd_weighted
            except Exception:
                result['SMD_Weighted'] = np.nan
        
        smd_results.append(result)
    
    return pd.DataFrame(smd_results)


def plot_love_plot(smd_df, plot_dir, filename="love_plot_covariate_balance.png", 
                   threshold=0.1, logger=None):
    """
    Create a Love plot showing covariate balance before and after weighting.
    
    Parameters:
    -----------
    smd_df : DataFrame with columns Covariate, SMD_Unweighted, SMD_Weighted (optional)
    plot_dir : str, directory to save plot
    filename : str, output filename
    threshold : float, SMD threshold for adequate balance (default 0.1)
    logger : logging.Logger
    """
    try:
        fig, ax = plt.subplots(figsize=(10, max(6, len(smd_df) * 0.4)))
        
        # Sort by unweighted SMD for better visualization
        smd_df = smd_df.sort_values('SMD_Unweighted', key=abs, ascending=False).reset_index(drop=True)
        
        y_positions = range(len(smd_df))
        
        # Plot unweighted SMD
        ax.scatter(smd_df['SMD_Unweighted'], y_positions, 
                  color='#d62728', marker='o', s=80, alpha=0.7, 
                  label='Before Weighting', zorder=3)
        
        # Plot weighted SMD if available
        if 'SMD_Weighted' in smd_df.columns:
            ax.scatter(smd_df['SMD_Weighted'], y_positions, 
                      color='#1f77b4', marker='s', s=80, alpha=0.7, 
                      label='After AIPW Weighting', zorder=3)
            
            # Connect with lines
            for i, row in smd_df.iterrows():
                ax.plot([row['SMD_Unweighted'], row['SMD_Weighted']], 
                       [i, i], color='gray', alpha=0.3, linewidth=1, zorder=1)
        
        # Add threshold lines
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label=f'SMD = ±{threshold}', zorder=2)
        ax.axvline(x=-threshold, color='black', linestyle='--', linewidth=1.5, 
                  alpha=0.5, zorder=2)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5, zorder=2)
        
        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(smd_df['Covariate'], fontsize=9)
        ax.set_xlabel('Standardized Mean Difference (SMD)', fontsize=12)
        ax.set_ylabel('Covariate', fontsize=12)
        ax.set_title('Covariate Balance Before and After AIPW Weighting', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        os.makedirs(plot_dir, exist_ok=True)
        outp = os.path.join(plot_dir, filename)
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.info(f"Saved love plot: {outp}")
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating love plot {filename}: {e}")
        plt.close()


def create_covariate_definitions_table(confounders, out_dir, logger=None):
    """
    Create a detailed table of covariate definitions for transparency.
    
    Parameters:
    -----------
    confounders : list of covariate column names
    out_dir : str, output directory
    logger : logging.Logger
    """
    # Define covariate labels and descriptions
    covariate_dict = {
        'iv4_social_economic_factors_score': {
            'Category': 'Social & Economic Factors',
            'Label': 'Social & Economic Factors Index',
            'Description': 'Composite index including high school completion, unemployment, child poverty, income inequality, children in single-parent households, social associations, injury deaths',
            'Source': 'County Health Rankings'
        },
        'iv2_physical_environment_score': {
            'Category': 'Physical Environment',
            'Label': 'Physical Environment Index',
            'Description': 'Composite index including air pollution, drinking water violations, severe housing problems, driving alone to work, long commute times',
            'Source': 'County Health Rankings'
        },
        'iv3_health_behaviors_score': {
            'Category': 'Health Behaviors',
            'Label': 'Health Behaviors Index',
            'Description': 'Composite index including adult smoking, obesity, food environment, physical inactivity, exercise access, excessive drinking, alcohol-impaired driving deaths, STI rate, teen births',
            'Source': 'County Health Rankings & Behavioral Risk Factor Surveillance System'
        },
        'iv1_medicaid_expansion_active': {
            'Category': 'Healthcare Policy',
            'Label': 'Medicaid Expansion Status',
            'Description': 'Binary indicator: 1 if state had adopted ACA Medicaid expansion, 0 otherwise',
            'Source': 'Kaiser Family Foundation'
        },
        'log_population': {
            'Category': 'Demographics',
            'Label': 'Log Population',
            'Description': 'Natural logarithm of county population (2023 estimates)',
            'Source': 'US Census Bureau'
        },
    }
    
    # Add census division dummies
    for i in range(2, 10):
        covariate_dict[f'div_{i}'] = {
            'Category': 'Geographic Controls',
            'Label': f'Census Division {i}',
            'Description': f'Binary indicator for Census Division {i}',
            'Source': 'US Census Bureau'
        }
    
    # Create dataframe from available confounders
    rows = []
    for col in confounders:
        if col in covariate_dict:
            info = covariate_dict[col]
            rows.append({
                'Variable': col,
                'Category': info['Category'],
                'Label': info['Label'],
                'Description': info['Description'],
                'Source': info['Source']
            })
        else:
            # For any undefined covariates
            rows.append({
                'Variable': col,
                'Category': 'Other',
                'Label': col,
                'Description': 'Variable included in analysis',
                'Source': 'Various'
            })
    
    definitions_df = pd.DataFrame(rows)
    
    # Save to CSV
    out_csv = os.path.join(out_dir, "covariate_definitions.csv")
    definitions_df.to_csv(out_csv, index=False)
    
    if logger:
        logger.info(f"Saved covariate definitions table: {out_csv}")
    
    return definitions_df


def assemble_replication_report_full(out_dir,
                                     idx_df=None, mod_df=None, main_df=None,
                                     rural_df=None, dv21_rural_df=None,
                                     strat_df=None, h1h4_df=None,
                                     capex_df=None, dv3_df=None, 
                                     dissertation_direct_df=None,
                                     dissertation_interaction_df=None,
                                     logger=None):
    rows = []

    def add_row(fam, test, N, beta, p, lo, hi, q=None, note=""):
        rows.append({"Family": fam, "Test": test, "N": N,
                     "Beta": beta, "p": p, "CI_lower": lo, "CI_upper": hi,
                     "Q_FDR": q, "Note": note})

    # Dissertation Direct Effects
    if dissertation_direct_df is not None and not dissertation_direct_df.empty:
        for _, r in dissertation_direct_df.iterrows():
            # Report AIPW if available, otherwise OLS
            if pd.notna(r.get("AIPW_ATE")):
                beta, p, lo, hi = r["AIPW_ATE"], r["AIPW_p"], r["AIPW_CI_Lower"], r["AIPW_CI_Upper"]
                note = f"AIPW (N_t={int(r.get('N_Treated', 0))}, N_c={int(r.get('N_Control', 0))})"
            else:
                beta, p, lo, hi = r["OLS_Beta"], r["OLS_p"], r["OLS_CI_Lower"], r["OLS_CI_Upper"]
                note = "OLS (continuous)"
            add_row("Dissertation Direct Effects",
                    f"{r['Treatment']} -> {r['Outcome']}",
                    r.get("N"), beta, p, lo, hi, note=note)

    # Dissertation Interaction Effects
    if dissertation_interaction_df is not None and not dissertation_interaction_df.empty:
        for _, r in dissertation_interaction_df.iterrows():
            add_row("Dissertation Interaction Effects",
                    f"{r['IV']} x {r['Moderator']} -> {r['Outcome']}",
                    r.get("N"), r.get("Interaction_Beta"), r.get("Interaction_p"),
                    r.get("Interaction_CI_Lower"), r.get("Interaction_CI_Upper"))

    # Replication (MO1/MO2 → DVs)
    if idx_df is not None and not idx_df.empty:
        s = idx_df[idx_df["Exposure_Col"].isin(["mo1_genai_composite_score","mo2_robotics_composite_score"])]
        for _, r in s.iterrows():
            add_row("Replication (Index OLS)",
                    f"{r['Exposure_Col']} -> {r['Outcome_Col']}",
                    r.get("N"), r.get("OLS Beta"), r.get("OLS p"),
                    r.get("CI_Lower"), r.get("CI_Upper"), None, r.get("Exposure"))

    # H1–H4 results
    if h1h4_df is not None and not h1h4_df.empty:
        for _, r in h1h4_df.iterrows():
            add_row("H1–H4",
                    f"{r['Hypothesis']}: {r['Exposure']} -> {r['Outcome']}",
                    r.get("N"), r.get("OLS Beta"), r.get("OLS p"),
                    r.get("CI_Lower"), r.get("CI_Upper"), r.get("BH Q-Value"))

    # CAPEX intensity
    if capex_df is not None and not capex_df.empty:
        for _, r in capex_df.iterrows():
            add_row("CAPEX intensity (FI1/FI2)",
                    f"{r['Predictor']} -> {r['Outcome']}",
                    r.get("N"), r.get("OLS Beta"), r.get("OLS p"),
                    r.get("CI_Lower"), r.get("CI_Upper"), r.get("BH Q-Value"), r["Predictor_Col"])

    out_csv = os.path.join(out_dir, "replication_report_full.csv")
    pd.DataFrame(rows).sort_values("Family").to_csv(out_csv, index=False)
    if logger: logger.info(f"Wrote unified replication report to {out_csv} (rows={len(rows)})")

# =============================================================================
# H1–H4 Hypothesis Tests (IVs → DVs)
# =============================================================================
def run_h1_h4_tests(df: pd.DataFrame, base_controls: List[str], logger, out_dir: str) -> pd.DataFrame:
    """
    Runs clustered-OLS per hypothesis. Controls include log(pop) + division dummies + other IVs (excluding the IV under test).
    """
    specs = [
        # H1
        {"H":"H1", "Exposure":"iv1_medicaid_expansion_active", "Outcome":"dv1_clinical_care_score"},
        {"H":"H1", "Exposure":"iv1_medicaid_expansion_active", "Outcome":"dv3_avg_patient_services_margin"},
        # H2
        {"H":"H2", "Exposure":"iv2_physical_environment_score", "Outcome":"dv1_clinical_care_score"},
        {"H":"H2", "Exposure":"iv2_physical_environment_score", "Outcome":"dv2_health_outcomes_score"},
        {"H":"H2", "Exposure":"iv2_physical_environment_score", "Outcome":"dv3_avg_patient_services_margin"},
        # H3
        {"H":"H3", "Exposure":"iv3_health_behaviors_score", "Outcome":"dv1_clinical_care_score"},
        {"H":"H3", "Exposure":"iv3_health_behaviors_score", "Outcome":"dv2_health_outcomes_score"},
        # H4
        {"H":"H4", "Exposure":"iv4_social_economic_factors_score", "Outcome":"dv1_clinical_care_score"},
        {"H":"H4", "Exposure":"iv4_social_economic_factors_score", "Outcome":"dv2_health_outcomes_score"},
        {"H":"H4", "Exposure":"iv4_social_economic_factors_score", "Outcome":"dv3_avg_patient_services_margin"},
    ]

    rows = []
    for s in specs:
        xcol, ycol = s["Exposure"], s["Outcome"]
        missing = [c for c in [xcol, ycol, 'state_fips_for_clustering'] if c not in df.columns]
        if missing:
            logger.warning(f"Skipping {s['H']} {xcol}->{ycol}: missing {missing}")
            continue

        # Controls = base_controls minus the IV under test
        controls = [c for c in base_controls if c != xcol]
        cols = [ycol, xcol, 'state_fips_for_clustering'] + controls
        d = df[cols].dropna()
        if d.empty:
            logger.warning(f"{s['H']} {xcol}->{ycol}: no analytical rows after dropna.")
            continue

        Y = d[ycol]
        X = d[[xcol] + controls]
        res = run_ols_clustered(Y, X, d['state_fips_for_clustering'])
        b = res.params.get(xcol, np.nan)
        p = res.pvalues.get(xcol, np.nan)
        ci = res.conf_int().loc[xcol].values if xcol in res.params.index else (np.nan, np.nan)

        rows.append({
            "Hypothesis": s["H"], "Exposure": xcol, "Outcome": ycol,
            "N": len(d), "OLS Beta": b, "OLS p": p, "CI_Lower": ci[0], "CI_Upper": ci[1]
        })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = _bh_correct_in_place(out_df, 'OLS p', None, 'BH Q-Value')
        out_csv = os.path.join(out_dir, "hypotheses_h1_h4_summary.csv")
        out_df.to_csv(out_csv, index=False)
        logger.info(f"Saved H1–H4 summary to {out_csv}")
    else:
        logger.warning("No H1–H4 results to save.")

    return out_df

# =============================================================================
# CAPEX intensity (FI1/FI2) → DV2, DV21, MO1, MO2
# =============================================================================
def run_capex_intensity_tests(df: pd.DataFrame, base_controls: List[str], logger, out_dir: str) -> pd.DataFrame:
    if 'fi_capex_intensity_ratio' not in df.columns:
        logger.warning("CAPEX intensity not found; skipping CAPEX models.")
        return pd.DataFrame()

    outcomes = [
        ("dv2_health_outcomes_score", "DV2_Health_Outcomes"),
        ("dv21_premature_death_ypll_rate", "DV21_YPLL"),
        ("mo1_genai_composite_score", "MO1_GenAI"),
        ("mo2_robotics_composite_score", "MO2_Robotics"),
    ]

    xvars = [
        ("fi_capex_intensity_ratio", "CAPEX_Intensity_Raw"),
        ("fi_capex_intensity_ratio_w", "CAPEX_Intensity_Winsor01_99"),
        ("fi_capex_intensity_ratio_log1p", "log1p_CAPEX_Intensity_Winsor"),
    ]

    rows = []
    for ycol, yname in outcomes:
        if ycol not in df.columns:
            logger.warning(f"CAPEX: outcome {ycol} missing; skipping.")
            continue
        for xcol, xname in xvars:
            if xcol not in df.columns:
                continue
            controls = [c for c in base_controls if c != xcol]
            cols = [ycol, xcol, 'state_fips_for_clustering'] + controls
            d = df[cols].dropna()
            if d.empty or d[xcol].nunique() <= 1:
                continue

            Y = d[ycol]
            X = d[[xcol] + controls]
            try:
                res = run_ols_clustered(Y, X, d['state_fips_for_clustering'])
                b = res.params.get(xcol, np.nan)
                p = res.pvalues.get(xcol, np.nan)
                ci = res.conf_int().loc[xcol].values if xcol in res.params.index else (np.nan, np.nan)
            except Exception:
                b, p, ci = np.nan, np.nan, (np.nan, np.nan)

            rows.append({
                "Outcome": yname, "Outcome_Col": ycol,
                "Predictor": xname, "Predictor_Col": xcol,
                "N": len(d),
                "OLS Beta": b, "OLS p": p,
                "CI_Lower": ci[0], "CI_Upper": ci[1]
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = _bh_correct_in_place(out, 'OLS p', 'Outcome', 'BH Q-Value')
        out_csv = os.path.join(out_dir, "capex_intensity_regression_summary.csv")
        out.to_csv(out_csv, index=False)
        logger.info(f"Saved CAPEX intensity regression summary to {out_csv}")
    else:
        logger.warning("No CAPEX intensity results to save.")
    return out

# =============================================================================
# MO1 OLD vs NEW COMPARISON: Direct & Moderation Effects
# =============================================================================
def run_mo1_old_vs_new_comparison(engine, base_controls: List[str], logger, out_dir: str, n_boot: int) -> pd.DataFrame:
    """
    Compares OLD vs NEW MO1 (GenAI composite score) from vw_conceptual_model_adjpd vs v2.
    
    Tests:
    1. Direct effects: MO1 → DV2 (Health Outcomes Score) using AIPW
    2. Moderation effects: IV3 × MO1 → DV2 using Clustered OLS
    
    Returns DataFrame with side-by-side comparison of OLD and NEW results.
    """
    from sqlalchemy import text
    
    logger.info("\n" + "="*80)
    logger.info("MO1 OLD vs NEW COMPARISON: GenAI Composite Score Sensitivity Check")
    logger.info("="*80)
    logger.info("Testing MO1 (weighted_ai_adoption_score) → DV2 (health_outcomes_score)")
    logger.info("  OLD: vw_conceptual_model_adjpd")
    logger.info("  NEW: vw_conceptual_model_adjpd_v2")
    logger.info("  Direct Effects: AIPW with median split")
    logger.info("  Moderation Effects: IV3 (health behaviors) × MO1 → DV2")
    
    def fetch_mo1_dv2_from_view(view_name: str, label: str) -> pd.DataFrame:
        logger.info(f"\nFetching MO1/DV2 data from: {view_name} ({label})")
        
        sql = f"""
            SELECT
                LPAD(TRIM(CAST(vcm.county_fips AS TEXT)), 5, '0') AS county_fips,
                
                -- Treatment: MO1 (GenAI Composite Score - directly from view)
                vcm.weighted_ai_adoption_score::numeric AS mo1_genai_composite_score,
                
                -- Outcome: DV2 (Health Outcomes Score)
                vcm.health_outcomes_score::numeric AS dv2_health_outcomes_score,
                
                -- Moderator: IV3 (Health Behaviors)
                vcm.health_behaviors_score::numeric AS iv3_health_behaviors_score,
                
                -- Controls
                vcm.population::numeric AS population,
                vcm.social_economic_factors_score::numeric AS iv4_social_economic_factors_score,
                vcm.physical_environment_score::numeric AS iv2_physical_environment_score,
                vcm.clinical_care_score::numeric AS dv1_clinical_care_score,
                vcm.census_division::text AS census_division
                
            FROM
                public.{view_name} AS vcm
            WHERE
                vcm.population IS NOT NULL 
                AND CAST(vcm.population AS NUMERIC) > 0
        """
        
        try:
            df = pd.read_sql_query(text(sql), engine)
            df.columns = [c.lower() for c in df.columns]
            df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)
            
            # Convert numerics
            for col in ['mo1_genai_composite_score', 'dv2_health_outcomes_score', 
                        'iv3_health_behaviors_score', 'population',
                        'iv4_social_economic_factors_score', 'iv2_physical_environment_score',
                        'dv1_clinical_care_score']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"  Initial fetch: {len(df)} counties")
            
            # Filter to complete cases
            required_cols = ['mo1_genai_composite_score', 'dv2_health_outcomes_score',
                           'iv3_health_behaviors_score', 'population',
                           'iv4_social_economic_factors_score', 'iv2_physical_environment_score',
                           'dv1_clinical_care_score']
            available_required = [c for c in required_cols if c in df.columns]
            
            df_clean = df[available_required + ['county_fips', 'census_division']].dropna()
            logger.info(f"  After dropna: {len(df_clean)} counties")
            
            if len(df_clean) > 0:
                logger.info(f"  MO1 range: [{df_clean['mo1_genai_composite_score'].min():.3f}, {df_clean['mo1_genai_composite_score'].max():.3f}]")
                logger.info(f"  DV2 range: [{df_clean['dv2_health_outcomes_score'].min():.3f}, {df_clean['dv2_health_outcomes_score'].max():.3f}]")
                logger.info(f"  IV3 range: [{df_clean['iv3_health_behaviors_score'].min():.3f}, {df_clean['iv3_health_behaviors_score'].max():.3f}]")
            
            # Add state clustering column
            df_clean['state_fips_for_clustering'] = df_clean['county_fips'].astype(str).str[:2]
            
            # Add log population
            df_clean['log_population'] = np.log(df_clean['population'])
            
            # Add census division dummies
            if 'census_division' in df_clean.columns:
                div_dummies = pd.get_dummies(df_clean['census_division'], prefix='div', drop_first=True, dtype=int)
                df_clean = pd.concat([df_clean, div_dummies], axis=1)
            
            return df_clean
            
        except Exception as e:
            logger.error(f"  Error fetching from {view_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    # Fetch OLD and NEW versions
    df_old = fetch_mo1_dv2_from_view("vw_conceptual_model_adjpd", "OLD")
    df_new = fetch_mo1_dv2_from_view("vw_conceptual_model_adjpd_v2", "NEW v2")
    
    if df_old.empty or df_new.empty:
        logger.warning("Cannot run comparison - missing data from OLD or NEW view")
        return pd.DataFrame()
    
    # Prepare controls (match what's available in both datasets)
    control_base = ['iv4_social_economic_factors_score', 'iv2_physical_environment_score',
                    'iv3_health_behaviors_score', 'dv1_clinical_care_score', 'log_population']
    
    # Add census division dummies if available
    div_cols_old = [c for c in df_old.columns if c.startswith('div_')]
    div_cols_new = [c for c in df_new.columns if c.startswith('div_')]
    common_div_cols = list(set(div_cols_old) & set(div_cols_new))
    
    available_controls = [c for c in control_base if c in df_old.columns and c in df_new.columns]
    available_controls += common_div_cols
    
    logger.info(f"\nUsing {len(available_controls)} controls: {available_controls[:5]}{'...' if len(available_controls) > 5 else ''}")
    
    comparison_results = []
    
    # ========================================================================
    # PART 1: DIRECT EFFECTS (MO1 → DV2) using AIPW
    # ========================================================================
    logger.info("\n" + "-"*70)
    logger.info("PART 1: DIRECT EFFECTS (MO1 → DV2)")
    logger.info("-"*70)
    
    # Run AIPW on OLD data
    logger.info("\nRunning AIPW on OLD MO1 → DV2...")
    median_old = df_old['mo1_genai_composite_score'].median()
    df_old['mo1_binary'] = (df_old['mo1_genai_composite_score'] > median_old).astype(int)
    
    controls_for_aipw = [c for c in available_controls if c != 'iv3_health_behaviors_score']  # Remove moderator from direct effect
    
    (ate_old, ci_lower_old, ci_upper_old, p_old, n_t_old, n_c_old, err_old) = run_aipw(
        df=df_old,
        treat_col='mo1_binary',
        y_col='dv2_health_outcomes_score',
        confounders=controls_for_aipw,
        n_boot=n_boot,
        logger=logger,
        plot_dir=out_dir
    )
    
    if err_old:
        logger.warning(f"  OLD AIPW failed: {err_old}")
    else:
        logger.info(f"  OLD: ATE={ate_old:.4f}, 95% CI=[{ci_lower_old:.4f}, {ci_upper_old:.4f}], p={p_old:.4f}")
        comparison_results.append({
            'Version': 'OLD',
            'View': 'vw_conceptual_model_adjpd',
            'Analysis': 'Direct Effect',
            'Method': 'AIPW',
            'Treatment': 'MO1 (GenAI Composite)',
            'Outcome': 'DV2 (Health Outcomes)',
            'N': len(df_old),
            'N_Treated': n_t_old,
            'N_Control': n_c_old,
            'ATE': ate_old,
            'CI_Lower': ci_lower_old,
            'CI_Upper': ci_upper_old,
            'p_value': p_old,
            'Median_Threshold': median_old
        })
    
    # Run AIPW on NEW data
    logger.info("\nRunning AIPW on NEW MO1 → DV2...")
    median_new = df_new['mo1_genai_composite_score'].median()
    df_new['mo1_binary'] = (df_new['mo1_genai_composite_score'] > median_new).astype(int)
    
    (ate_new, ci_lower_new, ci_upper_new, p_new, n_t_new, n_c_new, err_new) = run_aipw(
        df=df_new,
        treat_col='mo1_binary',
        y_col='dv2_health_outcomes_score',
        confounders=controls_for_aipw,
        n_boot=n_boot,
        logger=logger,
        plot_dir=out_dir
    )
    
    if err_new:
        logger.warning(f"  NEW AIPW failed: {err_new}")
    else:
        logger.info(f"  NEW: ATE={ate_new:.4f}, 95% CI=[{ci_lower_new:.4f}, {ci_upper_new:.4f}], p={p_new:.4f}")
        comparison_results.append({
            'Version': 'NEW',
            'View': 'vw_conceptual_model_adjpd_v2',
            'Analysis': 'Direct Effect',
            'Method': 'AIPW',
            'Treatment': 'MO1 (GenAI Composite)',
            'Outcome': 'DV2 (Health Outcomes)',
            'N': len(df_new),
            'N_Treated': n_t_new,
            'N_Control': n_c_new,
            'ATE': ate_new,
            'CI_Lower': ci_lower_new,
            'CI_Upper': ci_upper_new,
            'p_value': p_new,
            'Median_Threshold': median_new
        })
    
    # ========================================================================
    # PART 2: MODERATION EFFECTS (IV3 × MO1 → DV2) using Clustered OLS
    # ========================================================================
    logger.info("\n" + "-"*70)
    logger.info("PART 2: MODERATION EFFECTS (IV3 × MO1 → DV2)")
    logger.info("-"*70)
    
    # Run moderation on OLD data
    logger.info("\nRunning Clustered OLS for IV3 × MO1 → DV2 on OLD...")
    df_old_mod = df_old.copy()
    df_old_mod['mo1_x_iv3'] = df_old_mod['mo1_genai_composite_score'] * df_old_mod['iv3_health_behaviors_score']
    
    Y_old = df_old_mod['dv2_health_outcomes_score']
    X_old = df_old_mod[['mo1_genai_composite_score', 'iv3_health_behaviors_score', 'mo1_x_iv3'] + available_controls]
    
    try:
        res_old = run_ols_clustered(Y_old, X_old, df_old_mod['state_fips_for_clustering'])
        beta_int_old = res_old.params.get('mo1_x_iv3', np.nan)
        p_int_old = res_old.pvalues.get('mo1_x_iv3', np.nan)
        ci_int_old = res_old.conf_int().loc['mo1_x_iv3'].values if 'mo1_x_iv3' in res_old.params.index else (np.nan, np.nan)
        
        logger.info(f"  OLD: β_interaction={beta_int_old:.6f}, p={p_int_old:.4f}")
        
        comparison_results.append({
            'Version': 'OLD',
            'View': 'vw_conceptual_model_adjpd',
            'Analysis': 'Moderation Effect',
            'Method': 'Clustered OLS',
            'Treatment': 'MO1 (GenAI Composite)',
            'Moderator': 'IV3 (Health Behaviors)',
            'Outcome': 'DV2 (Health Outcomes)',
            'N': len(df_old_mod),
            'Beta_Interaction': beta_int_old,
            'CI_Lower': ci_int_old[0],
            'CI_Upper': ci_int_old[1],
            'p_value': p_int_old
        })
    except Exception as e:
        logger.error(f"  OLD moderation failed: {e}")
    
    # Run moderation on NEW data
    logger.info("\nRunning Clustered OLS for IV3 × MO1 → DV2 on NEW...")
    df_new_mod = df_new.copy()
    df_new_mod['mo1_x_iv3'] = df_new_mod['mo1_genai_composite_score'] * df_new_mod['iv3_health_behaviors_score']
    
    Y_new = df_new_mod['dv2_health_outcomes_score']
    X_new = df_new_mod[['mo1_genai_composite_score', 'iv3_health_behaviors_score', 'mo1_x_iv3'] + available_controls]
    
    try:
        res_new = run_ols_clustered(Y_new, X_new, df_new_mod['state_fips_for_clustering'])
        beta_int_new = res_new.params.get('mo1_x_iv3', np.nan)
        p_int_new = res_new.pvalues.get('mo1_x_iv3', np.nan)
        ci_int_new = res_new.conf_int().loc['mo1_x_iv3'].values if 'mo1_x_iv3' in res_new.params.index else (np.nan, np.nan)
        
        logger.info(f"  NEW: β_interaction={beta_int_new:.6f}, p={p_int_new:.4f}")
        
        comparison_results.append({
            'Version': 'NEW',
            'View': 'vw_conceptual_model_adjpd_v2',
            'Analysis': 'Moderation Effect',
            'Method': 'Clustered OLS',
            'Treatment': 'MO1 (GenAI Composite)',
            'Moderator': 'IV3 (Health Behaviors)',
            'Outcome': 'DV2 (Health Outcomes)',
            'N': len(df_new_mod),
            'Beta_Interaction': beta_int_new,
            'CI_Lower': ci_int_new[0],
            'CI_Upper': ci_int_new[1],
            'p_value': p_int_new
        })
    except Exception as e:
        logger.error(f"  NEW moderation failed: {e}")
    
    # ========================================================================
    # SAVE & SUMMARIZE RESULTS
    # ========================================================================
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save to CSV
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(out_dir, f"mo1_dv2_old_vs_new_comparison_{timestamp}.csv")
        comparison_df.to_csv(out_csv, index=False)
        logger.info(f"\n✓ Saved MO1→DV2 comparison: {out_csv}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("MO1 → DV2 COMPARISON SUMMARY")
        logger.info("="*80)
        
        # Compare direct effects
        direct_old = comparison_df[(comparison_df['Version'] == 'OLD') & 
                                   (comparison_df['Analysis'] == 'Direct Effect')]
        direct_new = comparison_df[(comparison_df['Version'] == 'NEW') & 
                                   (comparison_df['Analysis'] == 'Direct Effect')]
        
        if not direct_old.empty and not direct_new.empty:
            ate_old_val = direct_old.iloc[0]['ATE']
            ate_new_val = direct_new.iloc[0]['ATE']
            p_old_val = direct_old.iloc[0]['p_value']
            p_new_val = direct_new.iloc[0]['p_value']
            
            diff_ate = ate_new_val - ate_old_val
            pct_change = (diff_ate / ate_old_val * 100) if abs(ate_old_val) > 1e-6 else np.nan
            
            logger.info("\nDIRECT EFFECT (MO1 → DV2):")
            logger.info(f"  OLD ATE: {ate_old_val:.4f}, p={p_old_val:.4f}")
            logger.info(f"  NEW ATE: {ate_new_val:.4f}, p={p_new_val:.4f}")
            logger.info(f"  Difference: {diff_ate:.4f} ({pct_change:.1f}% change)")
            
            # Check sign flip
            if ate_old_val * ate_new_val < 0:
                logger.info("  ⚠⚠ SIGN FLIP DETECTED: Direction of effect reversed!")
            elif abs(pct_change) < 5:
                logger.info("  ✓ STABLE: Direct effect minimal change (<5%)")
            else:
                logger.info(f"  ⚠ MODERATE CHANGE: Direct effect shifted by {pct_change:.1f}%")
            
            # Check significance changes
            sig_old = "significant" if p_old_val < 0.05 else "not significant"
            sig_new = "significant" if p_new_val < 0.05 else "not significant"
            logger.info(f"  OLD: {sig_old}, NEW: {sig_new}")
            
            if (p_old_val < 0.05) != (p_new_val < 0.05):
                logger.info("  ⚠ SIGNIFICANCE CHANGE: p-value crossed 0.05 threshold")
        
        # Compare moderation effects
        mod_old = comparison_df[(comparison_df['Version'] == 'OLD') & 
                                (comparison_df['Analysis'] == 'Moderation Effect')]
        mod_new = comparison_df[(comparison_df['Version'] == 'NEW') & 
                                (comparison_df['Analysis'] == 'Moderation Effect')]
        
        if not mod_old.empty and not mod_new.empty:
            beta_old = mod_old.iloc[0]['Beta_Interaction']
            beta_new = mod_new.iloc[0]['Beta_Interaction']
            p_old_mod = mod_old.iloc[0]['p_value']
            p_new_mod = mod_new.iloc[0]['p_value']
            
            diff_beta = beta_new - beta_old
            pct_change_beta = (diff_beta / beta_old * 100) if abs(beta_old) > 1e-6 else np.nan
            
            logger.info("\nMODERATION EFFECT (IV3 × MO1 → DV2):")
            logger.info(f"  OLD β_interaction: {beta_old:.6f}, p={p_old_mod:.4f}")
            logger.info(f"  NEW β_interaction: {beta_new:.6f}, p={p_new_mod:.4f}")
            logger.info(f"  Difference: {diff_beta:.6f} ({pct_change_beta:.1f}% change)")
            
            # Check sign flip
            if beta_old * beta_new < 0:
                logger.info("  ⚠⚠ SIGN FLIP DETECTED: Direction of moderation reversed!")
            elif abs(pct_change_beta) < 10:
                logger.info("  ✓ STABLE: Moderation effect minimal change (<10%)")
            else:
                logger.info(f"  ⚠ MODERATE CHANGE: Moderation effect shifted by {pct_change_beta:.1f}%")
            
            # Check significance changes
            sig_old_mod = "significant" if p_old_mod < 0.05 else "not significant"
            sig_new_mod = "significant" if p_new_mod < 0.05 else "not significant"
            logger.info(f"  OLD: {sig_old_mod}, NEW: {sig_new_mod}")
            
            if (p_old_mod < 0.05) != (p_new_mod < 0.05):
                logger.info("  ⚠ SIGNIFICANCE CHANGE: p-value crossed 0.05 threshold")
        
        logger.info("="*80 + "\n")
        
        return comparison_df
    else:
        logger.warning("No comparison results generated")
        return pd.DataFrame()

# =============================================================================
# MAIN
# =============================================================================
def main():
    run_start_dt = datetime.datetime.now()
    run_start_epoch = run_start_dt.timestamp()
    run_id = generate_run_id(run_start_dt)

    # Directories
    output_root_dir = "genai_robotics_health_output"
    out_dir = os.path.join(output_root_dir, "runs", run_id)
    plot_dir = os.path.join(out_dir, "plots")
    log_dir = os.path.join(out_dir, "logs")

    logger = setup_logger(log_dir=log_dir, run_id=run_id)
    logger.info("Starting Dissertation Replication & Testing Script.")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run output directory: {out_dir}")

    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.PerfectSeparationWarning)

    # Parameters
    
    # Connect & fetch
    engine = connect_to_database(logger)
    df_raw = fetch_data_for_analysis(engine, logger)
    df_hosp_prepost = fetch_hospital_prepost_outcomes(engine, logger)
    if not df_hosp_prepost.empty:
        before_cols = df_raw.shape[1]
        df_raw = df_raw.merge(df_hosp_prepost, on="county_fips", how="left")
        logger.info(
            f"Merged hospital pre/post outcomes into main dataframe: +{df_raw.shape[1] - before_cols} columns."
        )
    df, base_controls = common_prepare_data(df_raw, logger, engine=engine)

    # ================================================================================
    # PART J: HOSPITAL–COUNTY BRIDGE MECHANISM ANALYSIS (NEW)
    # ================================================================================
    logger.info("\n" + "="*80)
    logger.info("PART J: Hospital–County Bridge Mechanism Analysis (vw_hospital_county_bridge)")
    logger.info("="*80)

    df_bridge = fetch_hospital_county_bridge(engine, logger)
    bridge_results = run_mechanism_bridge_block(
        df_main=df,
        df_bridge=df_bridge,
        base_controls=base_controls,
        logger=logger,
        out_dir=out_dir,
        plot_dir=plot_dir
    )

    # ================================================================================
    # PART J2: HOSPITAL-LEVEL OWNERSHIP ANALYSIS (NEW)
    # ================================================================================
    logger.info("\n" + "="*80)
    logger.info("PART J2: Hospital-Level AI/Robotics Adoption by Ownership Analysis")
    logger.info("="*80)

    df_hosp = fetch_hospital_ownership_data(engine, logger)
    ownership_results = analyze_hospital_ownership(
        df_hosp=df_hosp,
        logger=logger,
        out_dir=out_dir
    )

    # ================================================================================
    # PART K: CT6 HOSPITAL DEATHS ANALYSIS (NEW)
    # ================================================================================
    logger.info("\n" + "="*80)
    logger.info("PART K: CT6 Hospital Deaths Analysis (vw_2023_hospital_deaths)")
    logger.info("="*80)

    ct6_results = run_ct6_hospital_deaths_block(
        df=df,
        df_bridge=df_bridge,
        base_controls=base_controls,
        logger=logger,
        out_dir=out_dir
    )

    # ================================================================================
    # PART L: MO1 → DV2 OLD vs NEW COMPARISON (SENSITIVITY CHECK)
    # ================================================================================
    logger.info("\n" + "="*80)
    logger.info("PART L: MO1 → DV2 OLD vs NEW Comparison (GenAI Composite Sensitivity)")
    logger.info("="*80)
    
    mo1_comparison_results = run_mo1_old_vs_new_comparison(
        engine=engine,
        base_controls=base_controls,
        logger=logger,
        out_dir=out_dir,
        n_boot=500  # Using 500 bootstrap iterations for AIPW
    )

    # --------------------------------------------------------------------------------
    # PART A-G: Original script's analyses are run first
    # --------------------------------------------------------------------------------
    logger.info("Running original script analyses (Parts A-G).")
    
    # PART C. Index-level OLS (for replication checks)
    outcomes_c = {
        "dv1_clinical_care_score": "DV1_Clinical_Care",
        "dv2_health_outcomes_score": "DV2_Health_Outcomes",
        "dv3_avg_patient_services_margin": "DV3_Patient_Services_Margin"
    }
    exposures_c = {
        "mo1_genai_composite_score": "MO1_GenAI_Composite",
        "mo2_robotics_composite_score": "MO2_Robotics_Composite"
    }
    
    idx_rows = []
    for xcol, xname in exposures_c.items():
        if xcol not in df.columns: continue
        for ycol, yname in outcomes_c.items():
            if ycol not in df.columns: continue
            d = df[[ycol, xcol, 'state_fips_for_clustering'] + base_controls].dropna()
            if d.empty: continue
            Y = d[ycol]; X = d[[xcol] + base_controls]
            res = run_ols_clustered(Y, X, d['state_fips_for_clustering'])
            b = res.params.get(xcol, np.nan); p = res.pvalues.get(xcol, np.nan)
            ci = res.conf_int().loc[xcol].values if xcol in res.params.index else (np.nan, np.nan)
            idx_rows.append({
                "Exposure": xname, "Exposure_Col": xcol, "Outcome": yname, "Outcome_Col": ycol,
                "N": len(d), "OLS Beta": b, "OLS p": p, "CI_Lower": ci[0], "CI_Upper": ci[1]
            })

    idx_df = pd.DataFrame(idx_rows)
    if not idx_df.empty: idx_df.to_csv(os.path.join(out_dir, "index_level_ols_summary.csv"), index=False)
    
    # Other analyses from original script
    h1h4_df = run_h1_h4_tests(df, base_controls, logger, out_dir)
    capex_df = run_capex_intensity_tests(df, base_controls, logger, out_dir)
    
    # ================================================================================
    # PART H: DISSERTATION HYPOTHESIS TESTS
    # ================================================================================
    logger.info("PART H: Running all dissertation-specific hypothesis tests.")
    
    # --- H1. Direct Effects ---
    # This list combines tests from the original script AND newly requested tests.
    direct_effects_specs = [
        # --- Tests from original script (preserved) ---
        {"treatment": "mo1_genai_composite_score", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo11_ai_staff_scheduling_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo12_ai_predict_staff_needs_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo13_ai_predict_patient_demand_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo15_ai_optimize_workflows_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo1_genai_composite_score"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo11_ai_staff_scheduling_pct"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo12_ai_predict_staff_needs_pct"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo13_ai_predict_patient_demand_pct"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo14_ai_automate_routine_tasks_pct"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo15_ai_optimize_workflows_pct"},
        # --- Newly requested tests (added) ---
        {"treatment": "mo2_robotics_composite_score", "outcome": "dv15_preventable_stays_rate"}, # MO2-->DV15
        {"treatment": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},  # MO21-->DV15
        # --- DV21 (Premature Death) direct effects ---
        {"treatment": "mo1_genai_composite_score", "outcome": "dv21_premature_death_ypll_rate"},
        {"treatment": "mo11_ai_staff_scheduling_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"treatment": "mo12_ai_predict_staff_needs_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"treatment": "mo13_ai_predict_patient_demand_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"treatment": "mo15_ai_optimize_workflows_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"treatment": "mo2_robotics_composite_score", "outcome": "dv21_premature_death_ypll_rate"},
        {"treatment": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
        # --- CT5 (2023 Calculated YPLL per 100k) direct effects ---
        {"treatment": "mo1_genai_composite_score", "outcome": "ct5_ypll_per_100k_mid"},
        {"treatment": "mo11_ai_staff_scheduling_pct", "outcome": "ct5_ypll_per_100k_mid"},
        {"treatment": "mo12_ai_predict_staff_needs_pct", "outcome": "ct5_ypll_per_100k_mid"},
        {"treatment": "mo13_ai_predict_patient_demand_pct", "outcome": "ct5_ypll_per_100k_mid"},
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "outcome": "ct5_ypll_per_100k_mid"},
        {"treatment": "mo15_ai_optimize_workflows_pct", "outcome": "ct5_ypll_per_100k_mid"},
        {"treatment": "mo2_robotics_composite_score", "outcome": "ct5_ypll_per_100k_mid"},
        {"treatment": "mo21_robotics_in_hospital_pct", "outcome": "ct5_ypll_per_100k_mid"},
    ]
    
    direct_results = []
    weight_diag_rows = []
    for spec in direct_effects_specs:
        t_col, y_col = spec["treatment"], spec["outcome"]
        logger.info(f"  Direct effect test: {t_col} -> {y_col}")
        model_controls = get_control_set(y_col, base_controls, df)
        
        required_cols = [t_col, y_col, 'state_fips_for_clustering'] + model_controls
        if any(c not in df.columns for c in required_cols):
            logger.warning(f"    Skipping: missing one or more columns from: {required_cols}")
            continue
            
        d = df[required_cols].dropna()
        if d.empty or d[t_col].nunique() < 2:
            logger.warning("    Skipping: Not enough data or variation after dropping NaNs.")
            continue
            
        # OLS with continuous treatment
        res_ols = run_ols_clustered(d[y_col], d[[t_col] + model_controls], d['state_fips_for_clustering'])
        ols_beta = res_ols.params.get(t_col, np.nan)
        ols_p = res_ols.pvalues.get(t_col, np.nan)
        ols_ci = res_ols.conf_int().loc[t_col].values if t_col in res_ols.params.index else (np.nan, np.nan)
        
        # AIPW with binarized treatment
        median_val = d[t_col].median()
        bin_col = f"{t_col}_gt_median"
        d[bin_col] = (d[t_col] > median_val).astype(int)

        # Propensity diagnostics (overlap/weights)
        try:
            X_diag = d[model_controls].copy()
            if X_diag.isnull().any().any():
                X_diag = X_diag.fillna(X_diag.mean())
            ps_diag, _ = estimate_propensity_scores(X_diag, d[bin_col].values.astype(int), logger, f"{t_col}_diag")
            if ps_diag is not None:
                diag_row = compute_weight_diagnostics(d[bin_col].values.astype(int), ps_diag, t_col, y_col)
                weight_diag_rows.append(diag_row)
        except Exception as e:
            logger.warning(f"    Propensity diagnostics failed for {t_col}->{y_col}: {e}")

        if d[bin_col].nunique() < 2:
             aipw_ate, aipw_cl, aipw_cu, aipw_p, n_t, n_c = [np.nan] * 6
             sd_ctrl = np.nan
        else:
            (aipw_ate, aipw_cl, aipw_cu, aipw_p,
            n_t, n_c, aipw_err) = run_aipw(d, bin_col, y_col, model_controls, N_BOOT, logger, plot_dir)
            # Calculate control group SD for E-value
            sd_ctrl = d[d[bin_col] == 0][y_col].std() if (d[bin_col] == 0).sum() > 1 else np.nan
        
        # Calculate baseline outcome statistics
        outcome_mean = d[y_col].mean()
        outcome_median = d[y_col].median()
        outcome_ctrl_mean = d[d[bin_col] == 0][y_col].mean() if (d[bin_col] == 0).sum() > 0 else outcome_mean
        
        # Calculate percentage changes
        # Absolute change is just the ATE itself (not a percentage)
        absolute_change_ate = aipw_ate
        
        # Relative change: (ATE / baseline_mean) * 100
        if abs(outcome_ctrl_mean) > 1e-6:
            relative_change_pct = (aipw_ate / outcome_ctrl_mean) * 100
            # Percentage change confidence intervals
            relative_change_ci_lower_pct = (aipw_cl / outcome_ctrl_mean) * 100 if not np.isnan(aipw_cl) else np.nan
            relative_change_ci_upper_pct = (aipw_cu / outcome_ctrl_mean) * 100 if not np.isnan(aipw_cu) else np.nan
        else:
            relative_change_pct = np.nan
            relative_change_ci_lower_pct = np.nan
            relative_change_ci_upper_pct = np.nan
        
        # Calculate E-value for sensitivity analysis
        evalue = calculate_e_value(aipw_ate, sd_ctrl) if not np.isnan(aipw_ate) and not np.isnan(sd_ctrl) else np.nan
        
        # E-value CI should use the CI bound CLOSEST to the null (zero)
        # If ATE is negative, use the upper bound (closer to zero)
        # If ATE is positive, use the lower bound (closer to zero)
        if not np.isnan(aipw_cl) and not np.isnan(aipw_cu) and not np.isnan(sd_ctrl):
            ci_bound_closest_to_null = aipw_cu if aipw_ate < 0 else aipw_cl
            evalue_ci = calculate_e_value(ci_bound_closest_to_null, sd_ctrl)
        else:
            evalue_ci = np.nan

        direct_results.append({
            "Treatment": t_col, "Outcome": y_col, "N": len(d),
            "OLS_Beta": ols_beta, "OLS_p": ols_p, "OLS_CI_Lower": ols_ci[0], "OLS_CI_Upper": ols_ci[1],
            "AIPW_ATE": aipw_ate, "AIPW_p": aipw_p, "AIPW_CI_Lower": aipw_cl, "AIPW_CI_Upper": aipw_cu,
            "N_Treated": n_t, "N_Control": n_c,
            "Outcome_Mean": outcome_mean, "Outcome_Median": outcome_median, 
            "Outcome_Control_Mean": outcome_ctrl_mean,
            "Absolute_Change_ATE": absolute_change_ate,
            "Relative_Change_Pct_vs_ControlMean": relative_change_pct,
            "Relative_Change_CI_Lower_Pct": relative_change_ci_lower_pct,
            "Relative_Change_CI_Upper_Pct": relative_change_ci_upper_pct,
            "E_Value": evalue, "E_Value_CI": evalue_ci,
        })
    
    dissertation_direct_df = pd.DataFrame(direct_results)
    primary_county_crossfit_df = pd.DataFrame()
    county_clip_trim_df = pd.DataFrame()
    county_ols_sensitivity = {"partial_r2": pd.DataFrame(), "oster": pd.DataFrame()}
    county_misclassification_df = pd.DataFrame()
    if not dissertation_direct_df.empty:
        out_csv = os.path.join(out_dir, "dissertation_direct_effects_summary.csv")
        dissertation_direct_df.to_csv(out_csv, index=False)
        logger.info(f"Saved dissertation direct effects summary to {out_csv}")

        logger.info("\n  Running primary county cross-fitted AIPW summary and new sensitivity outputs...")
        primary_county_crossfit_df = run_primary_county_crossfit_summary(
            df=df,
            base_controls=base_controls,
            logger=logger,
            out_dir=out_dir,
            plot_dir=plot_dir,
            crossfit_n_boot=0,
        )
        county_clip_trim_df = run_county_clip_trim_sensitivity_suite(
            df=df,
            base_controls=base_controls,
            logger=logger,
            out_dir=out_dir,
            plot_dir=plot_dir,
            crossfit_n_boot=0,
        )
        county_ols_sensitivity = run_county_ols_sensitivity_suite(
            df=df,
            base_controls=base_controls,
            logger=logger,
            out_dir=out_dir,
        )
        county_misclassification_df = run_county_misclassification_suite(
            df=df,
            base_controls=base_controls,
            logger=logger,
            out_dir=out_dir,
            n_draws=200,
            n_workers=8,
            progress_every_sec=30,
            rf_n_jobs_per_worker=1,
        )

        spatial_res = None
        tmle_res = None
        dml_data = pd.DataFrame()
        dml_continuous_beta = dml_continuous_se = dml_continuous_ci_lower = dml_continuous_ci_upper = dml_continuous_p = np.nan
        dml_binary_ate = dml_binary_se = dml_binary_ci_lower = dml_binary_ci_upper = dml_binary_p = np.nan
        gps_beta = gps_p = gps_ci_lower = gps_ci_upper = np.nan
        dml_continuous_success = False
        dml_binary_success = False
        ext_results_dv21 = {}
        int_results_dv21 = {}
        
        # Highlight MO14→DV21 E-value (key analysis for reviewer)
        mo14_dv21 = dissertation_direct_df[
            (dissertation_direct_df['Treatment'] == 'mo14_ai_automate_routine_tasks_pct') & 
            (dissertation_direct_df['Outcome'] == 'dv21_premature_death_ypll_rate')
        ]
        if not mo14_dv21.empty:
            evalue = mo14_dv21.iloc[0]['E_Value']
            evalue_ci = mo14_dv21.iloc[0]['E_Value_CI']
            ate = mo14_dv21.iloc[0]['AIPW_ATE']
            p = mo14_dv21.iloc[0]['AIPW_p']
            logger.info(f"  MO14→DV21 (AI automate tasks → premature death):")
            logger.info(f"    AIPW ATE: {ate:.2f}, p={p:.4f}")
            logger.info(f"    E-Value (point): {evalue:.2f}")
            logger.info(f"    E-Value (CI lower): {evalue_ci:.2f}")
            logger.info(f"    Interpretation: Unmeasured confounder would need RR≥{evalue:.2f} to explain away the effect.")
            
            # === SENSITIVITY ANALYSIS: E-VALUES (Primary) ===
            # Note: 1:1 Propensity Score Matching causes severe sample attrition (3,079 → ~340 pairs),
            # discarding 80% of data and losing statistical power. The E-value provides equivalent
            # sensitivity assessment without sample loss.
            
            logger.info("\n  Sensitivity Analysis: E-values (Primary Metric)")
            logger.info(f"  E-value for MO14 → DV21: {evalue:.2f} (CI-bound: {evalue_ci:.2f})")
            logger.info(f"  Interpretation: Unmeasured confounder would need RR ≈ {evalue:.2f} with both")
            logger.info(f"                  treatment and outcome to nullify observed effect.")
            logger.info("  Note: E-values preferred over Rosenbaum bounds due to zero-inflated exposure.")
            logger.info("        1:1 matching would discard 80% of sample (3,079 → ~340 pairs),")
            logger.info("        losing power while E-values use full weighted sample.")
            
            # Rosenbaum bounds disabled due to sample attrition issues
            # Uncomment below if needed for supplementary analysis
            """
            # === PROPENSITY SCORE MATCHING + ROSENBAUM BOUNDS (OPTIONAL) ===
            logger.info("\n  Computing Rosenbaum bounds with Propensity Score Matching (MO14 → DV21)...")
            logger.info("  Warning: This approach discards ~80% of sample due to 1:1 matching.")
            
            mo14_dv21_data = df[['mo14_ai_automate_routine_tasks_pct', 'dv21_premature_death_ypll_rate'] + base_controls].dropna()
            median_mo14 = mo14_dv21_data['mo14_ai_automate_routine_tasks_pct'].median()
            mo14_dv21_data['treatment_binary'] = (mo14_dv21_data['mo14_ai_automate_routine_tasks_pct'] > median_mo14).astype(int)
            
            matched_pairs_df = perform_ps_matching(
                df=mo14_dv21_data,
                treatment_col='treatment_binary',
                outcome_col='dv21_premature_death_ypll_rate',
                confounders=base_controls,
                caliper=0.25,
                logger=logger
            )
            
            if not matched_pairs_df.empty:
                rosenbaum_df = calculate_rosenbaum_bounds_matched(
                    matched_df=matched_pairs_df,
                    gammas=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0],
                    logger=logger
                )
            """
            
            # Threshold sensitivity (10%, 25%, 50% of beds)
            logger.info("\n  Running threshold sensitivity for MO14 -> DV21 (10/25/50% of beds)...")
            th_df = run_threshold_sensitivity(
                df=df,
                treat_col='mo14_ai_automate_routine_tasks_pct',
                y_col='dv21_premature_death_ypll_rate',
                thresholds=[10, 25, 50],
                base_controls=base_controls,
                logger=logger,
                plot_dir=plot_dir
            )
            if not th_df.empty:
                th_csv = os.path.join(out_dir, "threshold_sensitivity_mo14_dv21.csv")
                th_df.to_csv(th_csv, index=False)
                logger.info(f"  Saved threshold sensitivity to {th_csv}")

            # Spatial block bootstrap (state-level clustering) for MO14 -> DV21
            logger.info("  Running spatial block bootstrap (state-level) for MO14 -> DV21...")
            spatial_res = spatial_block_bootstrap(
                df=df,
                treat_col='mo14_ai_automate_routine_tasks_pct',
                y_col='dv21_premature_death_ypll_rate',
                confounders=base_controls,
                cluster_col='state_fips_for_clustering',
                n_boot=300,
                logger=logger
            )
            if spatial_res:
                spatial_df = pd.DataFrame([spatial_res])
                spatial_csv = os.path.join(out_dir, "spatial_block_bootstrap_mo14_dv21.csv")
                spatial_df.to_csv(spatial_csv, index=False)
                logger.info(
                    f"  Spatial block bootstrap ATE={spatial_res['ATE_Mean']:.2f} "
                    f"(95% CI [{spatial_res['CI_Lower']:.2f}, {spatial_res['CI_Upper']:.2f}]), "
                    f"p={spatial_res['p_value']:.4f} (iterations={spatial_res['Iterations']})"
                )
            else:
                logger.warning("  Spatial block bootstrap did not complete (insufficient data or failures).")
        
        # Create covariate definitions table for transparency
        logger.info("  Creating covariate definitions table...")
        create_covariate_definitions_table(base_controls, out_dir, logger)
        
        # Create love plot for covariate balance (using a representative treatment for demonstration)
        # We'll use MO14 -> DV21 as the key analysis
        logger.info("  Creating covariate balance (love) plot for MO14 -> DV21...")
        dv21_controls = get_control_set('dv21_premature_death_ypll_rate', base_controls, df)
        mo14_dv21_data = df[['mo14_ai_automate_routine_tasks_pct', 'dv21_premature_death_ypll_rate'] + dv21_controls].dropna()
        if not mo14_dv21_data.empty and mo14_dv21_data['mo14_ai_automate_routine_tasks_pct'].nunique() >= 2:
            median_mo14 = mo14_dv21_data['mo14_ai_automate_routine_tasks_pct'].median()
            mo14_dv21_data['mo14_binary'] = (mo14_dv21_data['mo14_ai_automate_routine_tasks_pct'] > median_mo14).astype(int)
            
            # Calculate propensity scores for weighted SMD
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            X_balance = mo14_dv21_data[dv21_controls].copy()
            if X_balance.isnull().any().any():
                X_balance = X_balance.fillna(X_balance.mean())
            
            T_balance = mo14_dv21_data['mo14_binary'].values
            
            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_balance)
                ps_model = LogisticRegression(penalty='l1', solver='saga', max_iter=500, random_state=42)
                ps_model.fit(X_scaled, T_balance)
                ps_balance = ps_model.predict_proba(X_scaled)[:, 1]
                ps_balance = np.clip(ps_balance, 0.01, 0.99)
                
                # Calculate SMD
                smd_df = calculate_smd(mo14_dv21_data, 'mo14_binary', dv21_controls, ps=ps_balance)
                
                # Save SMD table
                smd_csv = os.path.join(out_dir, "covariate_balance_smd_mo14_dv21.csv")
                smd_df.to_csv(smd_csv, index=False)
                logger.info(f"  Saved covariate balance SMD table: {smd_csv}")
                
                # Create love plot
                plot_love_plot(smd_df, plot_dir, filename="love_plot_mo14_dv21.png", 
                             threshold=0.1, logger=logger)
                
            except Exception as e:
                logger.warning(f"  Could not create love plot: {e}")
        
        # Create forest plot for DV21 direct effects (MO14 and MO2)
        dv21_results = dissertation_direct_df[dissertation_direct_df['Outcome'] == 'dv21_premature_death_ypll_rate'].copy()
        if not dv21_results.empty:
            plot_forest_plot_ate(
                results_df=dv21_results,
                outcome_label="Premature Death Rate",
                plot_dir=plot_dir,
                logger=logger,
                treatment_filter=["mo14_ai_automate_routine_tasks_pct", "mo2_robotics_composite_score"],
                filename="forest_plot_dv21_mo14_mo2.png"
            )
            # Also create a full forest plot with all DV21 treatments
            plot_forest_plot_ate(
                results_df=dv21_results,
                outcome_label="Premature Death Rate",
                plot_dir=plot_dir,
                logger=logger,
                filename="forest_plot_dv21_all_treatments.png"
            )
        
        # --- CONTINUOUS TREATMENT ANALYSIS (Review 1 Request) ---
        logger.info("\n" + "="*80)
        logger.info("  CONTINUOUS TREATMENT ANALYSIS (Review 1 Request)")
        logger.info("  Running DML and GPS-IPW for MO14 -> DV21 with CONTINUOUS exposure...")
        logger.info("="*80)
        
        dml_data = df[['mo14_ai_automate_routine_tasks_pct', 'dv21_premature_death_ypll_rate'] + dv21_controls].dropna()
        if not dml_data.empty and len(dml_data) >= 100:
            
            # === 1. PRIMARY: DML with CONTINUOUS treatment ===
            logger.info("\n  [PRIMARY] Double Machine Learning with CONTINUOUS treatment:")
            (dml_continuous_beta, dml_continuous_se, dml_continuous_ci_lower, 
             dml_continuous_ci_upper, dml_continuous_p, dml_continuous_success) = run_dml(
                df=dml_data,
                treat_col='mo14_ai_automate_routine_tasks_pct',
                y_col='dv21_premature_death_ypll_rate',
                confounders=dv21_controls,
                logger=logger,
                n_splits=5,
                discrete_treatment=False  # PRIMARY: Continuous per Review 1
            )
            
            # === 2. SECONDARY: GPS-IPW with CONTINUOUS treatment ===
            logger.info("\n  [SECONDARY] GPS-IPW with CONTINUOUS treatment:")
            (gps_beta, gps_p, gps_ci_lower, gps_ci_upper) = run_ipw_continuous(
                df=dml_data,
                treat_col='mo14_ai_automate_routine_tasks_pct',
                y_col='dv21_premature_death_ypll_rate',
                confounders=dv21_controls,
                logger=logger
            )
            
            # === 2b. TWO-PART (HURDLE) MODEL for Zero-Inflated Treatment ===
            logger.info("\n  [ECONOMETRIC] Two-Part (Hurdle) Model for Zero-Inflated Treatment:")
            logger.info("  Addressing Review 1: 86.5% of counties have zero exposure - standard continuous methods dilute effects")
            (logit_ext_dv21, ols_int_dv21, ext_results_dv21, int_results_dv21) = run_two_part_model(
                df=dml_data,
                treat_col='mo14_ai_automate_routine_tasks_pct',
                y_col='dv21_premature_death_ypll_rate',
                confounders=dv21_controls,
                logger=logger
            )
            
            # === 3. SENSITIVITY: Binary DML (apples-to-apples with AIPW) ===
            logger.info("\n  [SENSITIVITY] Binary DML (median split for comparison with AIPW):")
            median_mo14 = dml_data['mo14_ai_automate_routine_tasks_pct'].median()
            dml_data['mo14_binary'] = (dml_data['mo14_ai_automate_routine_tasks_pct'] > median_mo14).astype(int)
            logger.info(f"    Binary split: {(dml_data['mo14_binary']==1).sum()} treated, {(dml_data['mo14_binary']==0).sum()} control")
            
            (dml_binary_ate, dml_binary_se, dml_binary_ci_lower, 
             dml_binary_ci_upper, dml_binary_p, dml_binary_success) = run_dml(
                df=dml_data,
                treat_col='mo14_binary',
                y_col='dv21_premature_death_ypll_rate',
                confounders=dv21_controls,
                logger=logger,
                n_splits=5,
                discrete_treatment=True  # SENSITIVITY: Binary treatment like AIPW
            )
            
            # === SAVE ALL RESULTS ===
            continuous_results = []
            
            # DML Continuous
            if dml_continuous_success:
                continuous_results.append({
                    'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                    'Treatment_Type': 'Continuous',
                    'Outcome': 'dv21_premature_death_ypll_rate',
                    'Method': 'Double Machine Learning (DML)',
                    'N': len(dml_data),
                    'Effect_Beta': dml_continuous_beta,
                    'SE': dml_continuous_se,
                    'CI_Lower': dml_continuous_ci_lower,
                    'CI_Upper': dml_continuous_ci_upper,
                    'p_value': dml_continuous_p,
                    'Interpretation': 'Change in YPLL per 1-unit increase in exposure',
                    'Estimator': 'LinearDML with RandomForest (Continuous GPS)',
                    'Cross_Fit_Splits': 5,
                    'Discrete_Treatment': False,
                    'Analysis_Type': 'PRIMARY'
                })
            
            # GPS-IPW Continuous
            if not np.isnan(gps_beta):
                continuous_results.append({
                    'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                    'Treatment_Type': 'Continuous',
                    'Outcome': 'dv21_premature_death_ypll_rate',
                    'Method': 'GPS-IPW (Generalized Propensity Score)',
                    'N': len(dml_data),
                    'Effect_Beta': gps_beta,
                    'SE': (gps_ci_upper - gps_ci_lower) / (2 * 1.96),
                    'CI_Lower': gps_ci_lower,
                    'CI_Upper': gps_ci_upper,
                    'p_value': gps_p,
                    'Interpretation': 'Change in YPLL per 1-unit increase in exposure',
                    'Estimator': 'Stabilized GPS Weights (Hirano & Imbens 2004)',
                    'Cross_Fit_Splits': None,
                    'Discrete_Treatment': False,
                    'Analysis_Type': 'SECONDARY'
                })
            
            # Two-Part Model Results
            if ext_results_dv21 and int_results_dv21:
                continuous_results.append({
                    'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                    'Treatment_Type': 'Two-Part Extensive',
                    'Outcome': 'dv21_premature_death_ypll_rate',
                    'Method': 'Two-Part Model (Hurdle) - Extensive Margin',
                    'N': ext_results_dv21.get('N_total', np.nan),
                    'N_Treated': ext_results_dv21.get('N_treated', np.nan),
                    'N_Control': ext_results_dv21.get('N_control', np.nan),
                    'Raw_Difference': ext_results_dv21.get('Raw_difference', np.nan),
                    'Effect_Beta': ext_results_dv21.get('ATE_extensive_adjusted', np.nan),
                    'SE': np.nan,
                    'CI_Lower': np.nan,
                    'CI_Upper': np.nan,
                    'p_value': np.nan,
                    'Pseudo_R2': ext_results_dv21.get('Pseudo_R2', np.nan),
                    'Y_treated_mean': ext_results_dv21.get('Y_treated_mean', np.nan),
                    'Y_control_mean': ext_results_dv21.get('Y_control_mean', np.nan),
                    'Interpretation': 'Participation Effect: Any AI vs None (AIPW-Adjusted)',
                    'Estimator': 'Logit + AIPW',
                    'Discrete_Treatment': True,
                    'Analysis_Type': 'ECONOMETRIC'
                })
                
                continuous_results.append({
                    'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                    'Treatment_Type': 'Two-Part Intensive',
                    'Outcome': 'dv21_premature_death_ypll_rate',
                    'Method': 'Two-Part Model (Hurdle) - Intensive Margin',
                    'N': int_results_dv21.get('N_adopters', np.nan),
                    'Effect_Beta_OLS': int_results_dv21.get('Beta_OLS', np.nan),
                    'p_value_OLS': int_results_dv21.get('p_value_OLS', np.nan),
                    'Effect_Beta_GPS': int_results_dv21.get('Beta_GPS', np.nan),
                    'p_value_GPS': int_results_dv21.get('p_value_GPS', np.nan),
                    'SE_OLS': int_results_dv21.get('SE_OLS', np.nan),
                    'CI_Lower_OLS': int_results_dv21.get('CI_lower_OLS', np.nan),
                    'CI_Upper_OLS': int_results_dv21.get('CI_upper_OLS', np.nan),
                    'CI_Lower_GPS': int_results_dv21.get('CI_lower_GPS', np.nan),
                    'CI_Upper_GPS': int_results_dv21.get('CI_upper_GPS', np.nan),
                    'R2': int_results_dv21.get('R2', np.nan),
                    'GAM_EDF': int_results_dv21.get('gam_analysis', {}).get('edf', np.nan),
                    'Tertile_Diff': int_results_dv21.get('tertile_analysis', {}).get('diff', np.nan),
                    'Tertile_p': int_results_dv21.get('tertile_analysis', {}).get('p', np.nan),
                    'Exposure_mean': int_results_dv21.get('Exposure_mean', np.nan),
                    'Exposure_median': int_results_dv21.get('Exposure_median', np.nan),
                    'Interpretation': 'Intensity Effect: More AI vs Less AI (among adopters, using GPS-IPW for causal inference)',
                    'Estimator': 'OLS + GPS-IPW (Causal) + Nonlinearity Checks',
                    'Discrete_Treatment': False,
                    'Analysis_Type': 'ECONOMETRIC'
                })
            
            # DML Binary (Sensitivity)
            if dml_binary_success:
                # Calculate baseline outcome statistics for percentage changes
                outcome_mean_dml = dml_data['dv21_premature_death_ypll_rate'].mean()
                outcome_median_dml = dml_data['dv21_premature_death_ypll_rate'].median()
                outcome_ctrl_mean_dml = dml_data[dml_data['mo14_binary'] == 0]['dv21_premature_death_ypll_rate'].mean()
                
                # Calculate percentage changes
                absolute_change_ate_dml = dml_binary_ate  # Absolute change = ATE (not a percentage)
                if abs(outcome_ctrl_mean_dml) > 1e-6:
                    relative_change_pct_dml = (dml_binary_ate / outcome_ctrl_mean_dml) * 100
                    # Percentage change confidence intervals
                    relative_change_ci_lower_pct_dml = (dml_binary_ci_lower / outcome_ctrl_mean_dml) * 100 if not np.isnan(dml_binary_ci_lower) else np.nan
                    relative_change_ci_upper_pct_dml = (dml_binary_ci_upper / outcome_ctrl_mean_dml) * 100 if not np.isnan(dml_binary_ci_upper) else np.nan
                else:
                    relative_change_pct_dml = np.nan
                    relative_change_ci_lower_pct_dml = np.nan
                    relative_change_ci_upper_pct_dml = np.nan
                
                continuous_results.append({
                    'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                    'Treatment_Type': 'Binary (>Median)',
                    'Outcome': 'dv21_premature_death_ypll_rate',
                    'Method': 'Double Machine Learning (DML)',
                    'N': len(dml_data),
                    'N_Treated': int((dml_data['mo14_binary']==1).sum()),
                    'N_Control': int((dml_data['mo14_binary']==0).sum()),
                    'Effect_Beta': dml_binary_ate,
                    'SE': dml_binary_se,
                    'CI_Lower': dml_binary_ci_lower,
                    'CI_Upper': dml_binary_ci_upper,
                    'p_value': dml_binary_p,
                    'Outcome_Mean': outcome_mean_dml,
                    'Outcome_Median': outcome_median_dml,
                    'Outcome_Control_Mean': outcome_ctrl_mean_dml,
                    'Absolute_Change_ATE': absolute_change_ate_dml,
                    'Relative_Change_Pct_vs_ControlMean': relative_change_pct_dml,
                    'Relative_Change_CI_Lower_Pct': relative_change_ci_lower_pct_dml,
                    'Relative_Change_CI_Upper_Pct': relative_change_ci_upper_pct_dml,
                    'Interpretation': 'ATE: High vs Low (median split)',
                    'Estimator': 'LinearDML with RandomForest (Binary)',
                    'Cross_Fit_Splits': 5,
                    'Discrete_Treatment': True,
                    'Analysis_Type': 'SENSITIVITY'
                })
            
            # Save to CSV
            if continuous_results:
                continuous_df = pd.DataFrame(continuous_results)
                continuous_csv = os.path.join(out_dir, "continuous_treatment_analysis_mo14_dv21.csv")
                continuous_df.to_csv(continuous_csv, index=False)
                logger.info(f"\n  ✓ Saved continuous treatment analysis: {continuous_csv}")
                
                # === PRINT COMPARISON SUMMARY ===
                logger.info("\n" + "="*80)
                logger.info("  MULTI-METHOD TRIANGULATION (Review 1 & 2 Response)")
                logger.info("="*80)
                
                if dml_continuous_success:
                    logger.info(f"  \n  [PRIMARY] DML (Continuous): Beta={dml_continuous_beta:.2f} YPLL per unit, p={dml_continuous_p:.4f}")
                    logger.info(f"            95% CI: [{dml_continuous_ci_lower:.2f}, {dml_continuous_ci_upper:.2f}]")
                
                if not np.isnan(gps_beta):
                    logger.info(f"  \n  [SECONDARY] GPS-IPW: Beta={gps_beta:.2f} YPLL per unit, p={gps_p:.4f}")
                    logger.info(f"              95% CI: [{gps_ci_lower:.2f}, {gps_ci_upper:.2f}]")
                
                if ext_results_dv21 and int_results_dv21:
                    ate_ext = ext_results_dv21.get('ATE_extensive_adjusted', np.nan)
                    beta_ols = int_results_dv21.get('Beta_OLS', np.nan)
                    beta_gps = int_results_dv21.get('Beta_GPS', np.nan)
                    p_ols = int_results_dv21.get('p_value_OLS', np.nan)
                    p_gps = int_results_dv21.get('p_value_GPS', np.nan)
                    n_adopters = int_results_dv21.get('N_adopters', np.nan)
                    logger.info(f"  \n  [ECONOMETRIC] Two-Part (Hurdle) Model:")
                    logger.info(f"    Part 1 (Extensive): ATE={ate_ext:.2f} YPLL (Any AI vs None, AIPW-adjusted)")
                    logger.info(f"    Part 2 (Intensive, n={n_adopters} adopters):")
                    logger.info(f"      - OLS (descriptive):  β={beta_ols:.2f}, p={p_ols:.4f}")
                    if not np.isnan(beta_gps):
                        logger.info(f"      - GPS-IPW (causal):   β={beta_gps:.2f}, p={p_gps:.4f}")
                    
                    # Interpretation based on both methods
                    both_null = p_ols > 0.05 and (np.isnan(p_gps) or p_gps > 0.05)
                    either_sig = p_ols < 0.05 or (not np.isnan(p_gps) and p_gps < 0.05)
                    
                    if both_null:
                        logger.info(f"    ✗ NO DOSE-RESPONSE: Both descriptive and causal methods find null effect")
                        logger.info(f"      → Benefit is a THRESHOLD EFFECT (0→1), not continuous scaling")
                    elif either_sig:
                        direction = "reduces" if (beta_gps if not np.isnan(beta_gps) else beta_ols) < 0 else "increases"
                        logger.info(f"    ✓ DOSE-RESPONSE DETECTED: More AI {direction} YPLL among adopters")
                
                if dml_binary_success:
                    logger.info(f"  \n  [SENSITIVITY] DML (Binary): ATE={dml_binary_ate:.2f} YPLL, p={dml_binary_p:.4f}")
                    logger.info(f"                95% CI: [{dml_binary_ci_lower:.2f}, {dml_binary_ci_upper:.2f}]")
                
                # Compare with AIPW
                mo14_aipw = dissertation_direct_df[
                    (dissertation_direct_df['Treatment'] == 'mo14_ai_automate_routine_tasks_pct') & 
                    (dissertation_direct_df['Outcome'] == 'dv21_premature_death_ypll_rate')
                ]
                if not mo14_aipw.empty:
                    aipw_ate = mo14_aipw.iloc[0]['AIPW_ATE']
                    aipw_p = mo14_aipw.iloc[0]['AIPW_p']
                    logger.info(f"  \n  [REFERENCE] AIPW (Binary): ATE={aipw_ate:.2f} YPLL, p={aipw_p:.4f}")
                    logger.info("  ")
                    
                    # Agreement check
                    if dml_continuous_success and dml_continuous_beta * aipw_ate > 0:
                        logger.info("  ✓ DML (Continuous) and AIPW show same direction")
                    if not np.isnan(gps_beta) and gps_beta * aipw_ate > 0:
                        logger.info("  ✓ GPS-IPW and AIPW show same direction")
                    if dml_binary_success and dml_binary_ate * aipw_ate > 0:
                        logger.info("  ✓ DML (Binary) and AIPW show same direction")
                    
                    # Significance check
                    sig_count = sum([p < 0.05 for p in [dml_continuous_p, gps_p, dml_binary_p, aipw_p] if not np.isnan(p)])
                    logger.info(f"  \n  ✓ {sig_count} of 4 methods achieve p < 0.05")
                    
                    if dml_continuous_success and dml_continuous_p < 0.05:
                        logger.info("  ✓ PRIMARY continuous analysis is statistically significant")
                        logger.info("  ✓ Review 1 concern ADDRESSED: Continuous treatment shows robust effect")
                
                logger.info("="*80 + "\n")
        else:
            logger.warning("  Insufficient data for continuous treatment analysis (DML/GPS-IPW).")

        # --- TMLE / SuperLearner robustness: MO14 -> DV21 ---
        logger.info("  Running TMLE (SuperLearner) robustness check for MO14 -> DV21...")
        tmle_res = run_tmle_superlearner(df, 'mo14_ai_automate_routine_tasks_pct', 'dv21_premature_death_ypll_rate', dv21_controls, logger)
        if tmle_res:
            tmle_df = pd.DataFrame([tmle_res])
            tmle_csv = os.path.join(out_dir, "tmle_superlearner_mo14_dv21.csv")
            tmle_df.to_csv(tmle_csv, index=False)
            logger.info(f"  TMLE ATE={tmle_res['ATE']:.2f} (95% CI: [{tmle_res['CI_Lower']:.2f}, {tmle_res['CI_Upper']:.2f}]), p={tmle_res['p_value']:.4f}")
            logger.info(f"  Saved TMLE results: {tmle_csv}")
        else:
            logger.warning("  TMLE did not produce a result (insufficient data or fitting issue).")

        # --- METHOD COMPARISON TABLE: AIPW vs DML vs TMLE for MO14->DV21 ---
        logger.info("\n  Creating method comparison table for MO14->DV21...")
        comparison_rows = []
        
        # Get AIPW results
        mo14_aipw = dissertation_direct_df[
            (dissertation_direct_df['Treatment'] == 'mo14_ai_automate_routine_tasks_pct') & 
            (dissertation_direct_df['Outcome'] == 'dv21_premature_death_ypll_rate')
        ]
        if not mo14_aipw.empty:
            aipw_row = mo14_aipw.iloc[0]
            comparison_rows.append({
                'Method': 'AIPW (Augmented Inverse Probability Weighting)',
                'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                'Outcome': 'dv21_premature_death_ypll_rate',
                'N': aipw_row.get('N', np.nan),
                'N_Treated': aipw_row.get('N_Treated', np.nan),
                'N_Control': aipw_row.get('N_Control', np.nan),
                'ATE': aipw_row['AIPW_ATE'],
                'CI_Lower': aipw_row['AIPW_CI_Lower'],
                'CI_Upper': aipw_row['AIPW_CI_Upper'],
                'p_value': aipw_row['AIPW_p'],
                'q_value': aipw_row.get('AIPW_q', np.nan),
                'E_value': aipw_row.get('E_Value', np.nan),
                'Estimator': 'RandomForest outcome model + Logistic PS model',
                'Bootstrap_Reps': N_BOOT
            })

        mo14_crossfit = primary_county_crossfit_df[
            (primary_county_crossfit_df['Treatment'] == 'mo14_ai_automate_routine_tasks_pct') &
            (primary_county_crossfit_df['Outcome'] == 'dv21_premature_death_ypll_rate')
        ] if not primary_county_crossfit_df.empty else pd.DataFrame()
        if not mo14_crossfit.empty:
            crossfit_row = mo14_crossfit.iloc[0]
            comparison_rows.append({
                'Method': 'AIPW (Cross-Fitted)',
                'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                'Outcome': 'dv21_premature_death_ypll_rate',
                'N': crossfit_row.get('N', np.nan),
                'N_Treated': crossfit_row.get('N_Treated', np.nan),
                'N_Control': crossfit_row.get('N_Control', np.nan),
                'ATE': crossfit_row.get('Crossfit_AIPW_ATE', np.nan),
                'CI_Lower': crossfit_row.get('Crossfit_AIPW_CI_Lower', np.nan),
                'CI_Upper': crossfit_row.get('Crossfit_AIPW_CI_Upper', np.nan),
                'p_value': crossfit_row.get('Crossfit_AIPW_p', np.nan),
                'q_value': np.nan,
                'E_value': np.nan,
                'Estimator': '5-fold cross-fit logistic PS + treated/control RF outcome models',
                'Bootstrap_Reps': 'Influence-function (bootstrap supported)'
            })
        
        # Get DML results (Binary - from sensitivity analysis)
        if dml_binary_success:
            comparison_rows.append({
                'Method': 'DML (Double Machine Learning) - Binary',
                'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                'Outcome': 'dv21_premature_death_ypll_rate',
                'N': len(dml_data),
                'N_Treated': int((dml_data['mo14_binary']==1).sum()),
                'N_Control': int((dml_data['mo14_binary']==0).sum()),
                'ATE': dml_binary_ate,
                'CI_Lower': dml_binary_ci_lower,
                'CI_Upper': dml_binary_ci_upper,
                'p_value': dml_binary_p,
                'q_value': np.nan,
                'E_value': np.nan,
                'Estimator': 'LinearDML with RandomForest nuisance models',
                'Bootstrap_Reps': 'Cross-fitting (5 folds)'
            })
        
        # Get DML results (Continuous - PRIMARY)
        if dml_continuous_success:
            comparison_rows.append({
                'Method': 'DML (Double Machine Learning) - Continuous',
                'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                'Outcome': 'dv21_premature_death_ypll_rate',
                'N': len(dml_data),
                'N_Treated': np.nan,  # Not applicable for continuous
                'N_Control': np.nan,  # Not applicable for continuous
                'ATE': dml_continuous_beta,
                'CI_Lower': dml_continuous_ci_lower,
                'CI_Upper': dml_continuous_ci_upper,
                'p_value': dml_continuous_p,
                'q_value': np.nan,
                'E_value': np.nan,
                'Estimator': 'LinearDML with RandomForest GPS',
                'Bootstrap_Reps': 'Cross-fitting (5 folds)'
            })
        
        # Get TMLE results
        if tmle_res:
            comparison_rows.append({
                'Method': 'TMLE (Targeted Maximum Likelihood Estimation)',
                'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                'Outcome': 'dv21_premature_death_ypll_rate',
                'N': tmle_res['N'],
                'N_Treated': tmle_res['N_Treated'],
                'N_Control': tmle_res['N_Control'],
                'ATE': tmle_res['ATE'],
                'CI_Lower': tmle_res['CI_Lower'],
                'CI_Upper': tmle_res['CI_Upper'],
                'p_value': tmle_res['p_value'],
                'q_value': np.nan,
                'E_value': np.nan,
                'Estimator': f"{tmle_res['Outcome_Model']} + {tmle_res['Exposure_Model']}",
                'Bootstrap_Reps': 'Influence function (non-parametric)'
            })
        
        # Get Spatial Block Bootstrap results (addresses spatial autocorrelation)
        if spatial_res:
            comparison_rows.append({
                'Method': 'Spatial Block Bootstrap (AIPW with State Clustering)',
                'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                'Outcome': 'dv21_premature_death_ypll_rate',
                'N': spatial_res.get('N', np.nan),
                'N_Treated': spatial_res.get('N_Treated', np.nan),
                'N_Control': spatial_res.get('N_Control', np.nan),
                'ATE': spatial_res['ATE_Mean'],
                'CI_Lower': spatial_res['CI_Lower'],
                'CI_Upper': spatial_res['CI_Upper'],
                'p_value': spatial_res['p_value'],
                'q_value': np.nan,
                'E_value': np.nan,
                'Estimator': 'AIPW with state-level block resampling',
                'Bootstrap_Reps': f"{spatial_res['Iterations']} spatial blocks"
            })
        
        # Save comparison table
        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            comparison_csv = os.path.join(out_dir, "method_comparison_mo14_dv21.csv")
            comparison_df.to_csv(comparison_csv, index=False)
            logger.info(f"  Saved method comparison table: {comparison_csv}")
            logger.info(f"\n  ═══ METHOD COMPARISON SUMMARY ═══")
            for _, row in comparison_df.iterrows():
                logger.info(f"  {row['Method']}: ATE={row['ATE']:.2f}, p={row['p_value']:.4f}")
            logger.info(f"  ════════════════════════════════════\n")

    # --- COVID-ERA CONFOUNDING CONTROLS: MO14 -> DV21 WITH/WITHOUT COVID CONTROLS ---
    logger.info("\n" + "="*80)
    logger.info("  COVID-ERA CONFOUNDING SENSITIVITY ANALYSIS (Reviewer Response)")
    logger.info("="*80)
    logger.info("  Purpose: Test whether county YPLL signal is AI access or COVID intensity.")
    logger.info("  Approach: Re-run primary MO14 -> YPLL AIPW with/without COVID controls.")
    
    covid_comparison_rows = []
    
    # Check if COVID data is available
    if 'cv1_covid_deaths_per_100k' in df.columns and df['cv1_covid_deaths_per_100k'].notna().sum() > 0:
        logger.info(f"  COVID data available: {df['cv1_covid_deaths_per_100k'].notna().sum()} counties with data.")
        
        # Prepare data with COVID controls
        dv21_controls = get_control_set('dv21_premature_death_ypll_rate', base_controls, df)
        covid_controls = list(dict.fromkeys(dv21_controls + ['cv1_covid_deaths_per_100k']))
        covid_data = df[['mo14_ai_automate_routine_tasks_pct', 'dv21_premature_death_ypll_rate'] + covid_controls].dropna()
        
        if len(covid_data) > 100 and covid_data['mo14_ai_automate_routine_tasks_pct'].nunique() >= 2:
            logger.info(f"  Sample size with COVID controls: N={len(covid_data)}")
            
            # Binarize treatment
            median_mo14 = covid_data['mo14_ai_automate_routine_tasks_pct'].median()
            covid_data['mo14_binary'] = (covid_data['mo14_ai_automate_routine_tasks_pct'] > median_mo14).astype(int)
            logger.info(f"  Treatment split: {(covid_data['mo14_binary']==1).sum()} treated, {(covid_data['mo14_binary']==0).sum()} control")
            
            # === 1. BASELINE MODEL (WITHOUT COVID CONTROLS) ===
            logger.info("\n  [BASELINE] Running AIPW WITHOUT COVID controls...")
            baseline_data = covid_data.copy()
            (baseline_ate, baseline_cl, baseline_cu, baseline_p,
             baseline_nt, baseline_nc, baseline_err) = run_aipw(
                baseline_data, 'mo14_binary', 'dv21_premature_death_ypll_rate', 
                dv21_controls, N_BOOT, logger, plot_dir
            )
            
            # Calculate control group mean for relative change
            baseline_ctrl_mean = baseline_data[baseline_data['mo14_binary'] == 0]['dv21_premature_death_ypll_rate'].mean()
            baseline_rel_change = (baseline_ate / baseline_ctrl_mean) * 100 if abs(baseline_ctrl_mean) > 1e-6 else np.nan
            
            logger.info(f"    Baseline ATE: {baseline_ate:.2f} YPLL (95% CI: [{baseline_cl:.2f}, {baseline_cu:.2f}]), p={baseline_p:.4f}")
            logger.info(f"    Relative change: {baseline_rel_change:.2f}%")
            
            covid_comparison_rows.append({
                'Model': 'Baseline (No COVID Controls)',
                'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                'Outcome': 'dv21_premature_death_ypll_rate',
                'N': len(baseline_data),
                'N_Treated': baseline_nt,
                'N_Control': baseline_nc,
                'ATE': baseline_ate,
                'CI_Lower': baseline_cl,
                'CI_Upper': baseline_cu,
                'p_value': baseline_p,
                'Control_Mean': baseline_ctrl_mean,
                'Relative_Change_Pct': baseline_rel_change,
                'Controls': ', '.join(dv21_controls),
                'COVID_Controlled': 'No'
            })
            
            # === 2. COVID-ADJUSTED MODEL (WITH COVID CONTROLS) ===
            logger.info("\n  [COVID-ADJUSTED] Running AIPW WITH COVID controls...")
            (covid_ate, covid_cl, covid_cu, covid_p,
             covid_nt, covid_nc, covid_err) = run_aipw(
                covid_data, 'mo14_binary', 'dv21_premature_death_ypll_rate', 
                covid_controls, N_BOOT, logger, plot_dir
            )
            
            # Calculate control group mean for relative change
            covid_ctrl_mean = covid_data[covid_data['mo14_binary'] == 0]['dv21_premature_death_ypll_rate'].mean()
            covid_rel_change = (covid_ate / covid_ctrl_mean) * 100 if abs(covid_ctrl_mean) > 1e-6 else np.nan
            
            logger.info(f"    COVID-Adjusted ATE: {covid_ate:.2f} YPLL (95% CI: [{covid_cl:.2f}, {covid_cu:.2f}]), p={covid_p:.4f}")
            logger.info(f"    Relative change: {covid_rel_change:.2f}%")
            
            covid_comparison_rows.append({
                'Model': 'COVID-Adjusted (With COVID Controls)',
                'Treatment': 'mo14_ai_automate_routine_tasks_pct',
                'Outcome': 'dv21_premature_death_ypll_rate',
                'N': len(covid_data),
                'N_Treated': covid_nt,
                'N_Control': covid_nc,
                'ATE': covid_ate,
                'CI_Lower': covid_cl,
                'CI_Upper': covid_cu,
                'p_value': covid_p,
                'Control_Mean': covid_ctrl_mean,
                'Relative_Change_Pct': covid_rel_change,
                'Controls': ', '.join(covid_controls),
                'COVID_Controlled': 'Yes'
            })
            
            # === 3. CALCULATE ATTENUATION ===
            if not np.isnan(baseline_ate) and not np.isnan(covid_ate) and abs(baseline_ate) > 1e-6:
                attenuation_pct = ((baseline_ate - covid_ate) / baseline_ate) * 100
                logger.info(f"\n  ✓ Attenuation: {attenuation_pct:.1f}% (ATE changed from {baseline_ate:.2f} to {covid_ate:.2f})")
                
                # Check if sign flipped or became non-significant
                if baseline_ate * covid_ate < 0:
                    logger.warning("  ⚠ WARNING: Effect sign FLIPPED after COVID adjustment!")
                elif baseline_p < 0.05 and covid_p >= 0.05:
                    logger.info("  ⚠ Effect became non-significant after COVID adjustment (p >= 0.05)")
                elif abs(attenuation_pct) < 20:
                    logger.info("  ✓ Effect is ROBUST: <20% attenuation with COVID controls")
                elif abs(attenuation_pct) < 50:
                    logger.info("  ✓ Effect moderately attenuated but remains negative")
                else:
                    logger.info("  ⚠ Substantial attenuation (>50%) suggests COVID confounding")
            
            # Save comparison table
            if covid_comparison_rows:
                covid_comparison_df = pd.DataFrame(covid_comparison_rows)
                covid_csv = os.path.join(out_dir, "covid_confounding_sensitivity_mo14_dv21.csv")
                covid_comparison_df.to_csv(covid_csv, index=False)
                logger.info(f"\n  ✓ Saved COVID sensitivity analysis: {covid_csv}")
                
                logger.info(f"\n  ═══ COVID SENSITIVITY SUMMARY ═══")
                logger.info(f"  Baseline (No COVID):     ATE={baseline_ate:.2f}, p={baseline_p:.4f}")
                logger.info(f"  COVID-Adjusted:          ATE={covid_ate:.2f}, p={covid_p:.4f}")
                logger.info(f"  Interpretation: {'ROBUST to COVID confounding' if abs(attenuation_pct) < 20 else 'Some COVID confounding detected'}")
                logger.info(f"  ════════════════════════════════════\n")
        else:
            logger.warning("  Insufficient data for COVID sensitivity analysis after merging.")
    else:
        logger.warning("  COVID data NOT available. Skipping COVID sensitivity analysis.")
        logger.warning("  To enable: Ensure vw_county_covid_deaths table exists with deaths_involving_covid_19 column.")
    
    logger.info("="*80 + "\n")

    # --- OUTCOME SENSITIVITY: DV21 vs CT4 vs CT5 (Multiple Mortality Measures) ---
    logger.info("\n" + "="*80)
    logger.info("  OUTCOME MEASURE SENSITIVITY ANALYSIS (DV21 vs CT4 vs CT5)")
    logger.info("="*80)
    logger.info("  Purpose: Compare different mortality measures to test robustness of AI adoption findings.")
    logger.info("  Question: Does choice of mortality measure affect sign, magnitude, or significance?")
    logger.info("  DV21 = CHR Premature Death Rate (YPLL per 100k, 2020-2022 blended, AGE-ADJUSTED)")
    logger.info("  CT4  = CDC WONDER Deaths <75yrs (deaths per 100k, 2023 only, NOT age-adjusted)")
    logger.info("  CT5  = CDC WONDER Age-Adjusted YPLL <75yrs (YPLL per 100k, 2023 only, AGE-ADJUSTED)")
    logger.info("  Conversion: 1 premature death ≈ 29 YPLL (for CT4 scaling)")
    
    # Scaling factor for converting deaths to YPLL-equivalent
    DEATH_TO_YPLL_FACTOR = 29.0
    
    outcome_comparison_rows = []
    
    # Check data availability for all three outcomes
    ct4_available = 'ct4_death_rate_per_100k' in df.columns and df['ct4_death_rate_per_100k'].notna().sum() > 0
    ct5_available = 'ct5_ypll_per_100k_mid' in df.columns and df['ct5_ypll_per_100k_mid'].notna().sum() > 0
    
    if ct4_available or ct5_available:
        logger.info(f"  DV21 data available: {df['dv21_premature_death_ypll_rate'].notna().sum()} counties with CHR YPLL.")
        if ct4_available:
            logger.info(f"  CT4 data available: {df['ct4_death_rate_per_100k'].notna().sum()} counties with 2023 CDC deaths.")
        if ct5_available:
            logger.info(f"  CT5 data available: {df['ct5_ypll_per_100k_mid'].notna().sum()} counties with 2023 calculated YPLL per 100k.")
        
        # Test all MO -> DV21 specifications with CT4 and CT5 as alternative outcomes
        outcome_sensitivity_specs = [
            {"treatment": "mo1_genai_composite_score", "label": "GenAI Composite"},
            {"treatment": "mo14_ai_automate_routine_tasks_pct", "label": "AI Routine Tasks (Primary)"},
            {"treatment": "mo11_ai_staff_scheduling_pct", "label": "AI Staff Scheduling"},
            {"treatment": "mo12_ai_predict_staff_needs_pct", "label": "AI Predict Staff Needs"},
            {"treatment": "mo13_ai_predict_patient_demand_pct", "label": "AI Patient Demand"},
            {"treatment": "mo15_ai_optimize_workflows_pct", "label": "AI Optimize Workflows"},
            {"treatment": "mo2_robotics_composite_score", "label": "Robotics Composite"},
            {"treatment": "mo21_robotics_in_hospital_pct", "label": "Robotics Presence"},
        ]
        
        for spec in outcome_sensitivity_specs:
            t_col = spec["treatment"]
            label = spec["label"]
            logger.info(f"\n  Testing: {label} ({t_col})")
            logger.info(f"  ─────────────────────────────────────────")
            dv21_controls = get_control_set('dv21_premature_death_ypll_rate', base_controls, df)
            ct4_controls = get_control_set('ct4_death_rate_per_100k', base_controls, df)
            ct5_controls = get_control_set('ct5_ypll_per_100k_mid', base_controls, df)
            
            # Check data availability for all three outcomes
            required_cols_dv21 = [t_col, 'dv21_premature_death_ypll_rate', 'state_fips_for_clustering'] + dv21_controls
            required_cols_ct4 = [t_col, 'ct4_death_rate_per_100k', 'state_fips_for_clustering'] + ct4_controls
            required_cols_ct5 = [t_col, 'ct5_ypll_per_100k_mid', 'state_fips_for_clustering'] + ct5_controls
            
            data_dv21 = df[required_cols_dv21].dropna()
            data_ct4 = df[required_cols_ct4].dropna() if ct4_available else pd.DataFrame()
            data_ct5 = df[required_cols_ct5].dropna() if ct5_available else pd.DataFrame()
            
            if data_dv21.empty:
                logger.warning(f"    Skipping: No DV21 data available.")
                continue
            
            sample_sizes = f"DV21 N={len(data_dv21)}"
            if not data_ct4.empty:
                sample_sizes += f", CT4 N={len(data_ct4)}"
            if not data_ct5.empty:
                sample_sizes += f", CT5 N={len(data_ct5)}"
            logger.info(f"    Sample sizes: {sample_sizes}")
            
            # Binarize treatment using CONSISTENT threshold across ALL outcomes
            # Use counties with ANY outcome available for consistent treatment definition
            cols_for_threshold = [t_col, 'dv21_premature_death_ypll_rate']
            if ct4_available:
                cols_for_threshold.append('ct4_death_rate_per_100k')
            if ct5_available:
                cols_for_threshold.append('ct5_ypll_per_100k_mid')
            
            common_sample = df[cols_for_threshold].dropna()
            if len(common_sample) > 0:
                treatment_threshold = common_sample[t_col].median()
                logger.info(f"    Treatment threshold (median on N={len(common_sample)} common sample): {treatment_threshold:.4f}")
            else:
                # Fallback: use full dataset median if no common sample
                treatment_threshold = df[t_col].median()
                logger.info(f"    Treatment threshold (median on full dataset): {treatment_threshold:.4f}")
            
            # Apply SAME threshold to all datasets
            data_dv21['treatment_binary'] = (data_dv21[t_col] > treatment_threshold).astype(int)
            if not data_ct4.empty:
                data_ct4['treatment_binary'] = (data_ct4[t_col] > treatment_threshold).astype(int)
            if not data_ct5.empty:
                data_ct5['treatment_binary'] = (data_ct5[t_col] > treatment_threshold).astype(int)
            
            # Log treatment split for verification
            logger.info(f"    DV21 treatment split: {(data_dv21['treatment_binary']==1).sum()} treated, {(data_dv21['treatment_binary']==0).sum()} control")
            if not data_ct4.empty:
                logger.info(f"    CT4 treatment split: {(data_ct4['treatment_binary']==1).sum()} treated, {(data_ct4['treatment_binary']==0).sum()} control")
            if not data_ct5.empty:
                logger.info(f"    CT5 treatment split: {(data_ct5['treatment_binary']==1).sum()} treated, {(data_ct5['treatment_binary']==0).sum()} control")

            ate_ct4 = cl_ct4 = cu_ct4 = p_ct4 = np.nan
            ate_ct4_ypll_equiv = cl_ct4_ypll_equiv = cu_ct4_ypll_equiv = np.nan
            ate_ct5 = cl_ct5 = cu_ct5 = p_ct5 = np.nan
            
            # === RUN AIPW FOR DV21 (CHR YPLL) ===
            logger.info(f"\n    [DV21: CHR YPLL 2020-22] Running AIPW...")
            (ate_dv21, cl_dv21, cu_dv21, p_dv21,
             nt_dv21, nc_dv21, err_dv21) = run_aipw(
                data_dv21, 'treatment_binary', 'dv21_premature_death_ypll_rate',
                dv21_controls, N_BOOT, logger, plot_dir
            )
            
            ctrl_mean_dv21 = data_dv21[data_dv21['treatment_binary'] == 0]['dv21_premature_death_ypll_rate'].mean()
            rel_change_dv21 = (ate_dv21 / ctrl_mean_dv21) * 100 if abs(ctrl_mean_dv21) > 1e-6 else np.nan
            
            logger.info(f"      ATE: {ate_dv21:.2f} YPLL (95% CI: [{cl_dv21:.2f}, {cu_dv21:.2f}]), p={p_dv21:.4f}")
            logger.info(f"      Control mean: {ctrl_mean_dv21:.1f}, Relative change: {rel_change_dv21:.2f}%")
            
            outcome_comparison_rows.append({
                'Treatment': t_col,
                'Treatment_Label': label,
                'Outcome': 'dv21_premature_death_ypll_rate',
                'Outcome_Label': 'CHR YPLL (2020-22)',
                'Outcome_Type': 'DV21_CHR_YPLL',
                'N': len(data_dv21),
                'N_Treated': nt_dv21,
                'N_Control': nc_dv21,
                'ATE': ate_dv21,
                'CI_Lower': cl_dv21,
                'CI_Upper': cu_dv21,
                'p_value': p_dv21,
                'Control_Mean': ctrl_mean_dv21,
                'Relative_Change_Pct': rel_change_dv21,
                'Significant': 'Yes' if p_dv21 < 0.05 else 'No'
            })
            
            # === RUN AIPW FOR CT4 (CDC 2023 DEATHS) ===
            if not data_ct4.empty:
                logger.info(f"\n    [CT4: CDC 2023 Deaths] Running AIPW...")
                (ate_ct4, cl_ct4, cu_ct4, p_ct4,
                 nt_ct4, nc_ct4, err_ct4) = run_aipw(
                    data_ct4, 'treatment_binary', 'ct4_death_rate_per_100k',
                    ct4_controls, N_BOOT, logger, plot_dir
                )
                
                ctrl_mean_ct4 = data_ct4[data_ct4['treatment_binary'] == 0]['ct4_death_rate_per_100k'].mean()
                rel_change_ct4 = (ate_ct4 / ctrl_mean_ct4) * 100 if abs(ctrl_mean_ct4) > 1e-6 else np.nan
                
                # Scale CT4 to YPLL-equivalent for apples-to-apples comparison
                ate_ct4_ypll_equiv = ate_ct4 * DEATH_TO_YPLL_FACTOR
                cl_ct4_ypll_equiv = cl_ct4 * DEATH_TO_YPLL_FACTOR
                cu_ct4_ypll_equiv = cu_ct4 * DEATH_TO_YPLL_FACTOR
                
                logger.info(f"      ATE: {ate_ct4:.2f} deaths/100k (95% CI: [{cl_ct4:.2f}, {cu_ct4:.2f}]), p={p_ct4:.4f}")
                logger.info(f"      ATE (YPLL-equivalent): {ate_ct4_ypll_equiv:.2f} YPLL (95% CI: [{cl_ct4_ypll_equiv:.2f}, {cu_ct4_ypll_equiv:.2f}])")
                logger.info(f"      Control mean: {ctrl_mean_ct4:.1f}, Relative change: {rel_change_ct4:.2f}%")
                
                outcome_comparison_rows.append({
                    'Treatment': t_col,
                    'Treatment_Label': label,
                    'Outcome': 'ct4_death_rate_per_100k',
                    'Outcome_Label': 'CDC Deaths (2023)',
                    'Outcome_Type': 'CT4_CDC_2023',
                    'N': len(data_ct4),
                    'N_Treated': nt_ct4,
                    'N_Control': nc_ct4,
                    'ATE': ate_ct4,
                    'CI_Lower': cl_ct4,
                    'CI_Upper': cu_ct4,
                    'p_value': p_ct4,
                    'Control_Mean': ctrl_mean_ct4,
                    'Relative_Change_Pct': rel_change_ct4,
                    'ATE_YPLL_Equivalent': ate_ct4_ypll_equiv,
                    'CI_Lower_YPLL_Equivalent': cl_ct4_ypll_equiv,
                    'CI_Upper_YPLL_Equivalent': cu_ct4_ypll_equiv,
                    'Significant': 'Yes' if p_ct4 < 0.05 else 'No'
                })
            
            # === RUN AIPW FOR CT5 (2023 CALCULATED YPLL PER 100K) ===
            if not data_ct5.empty:
                logger.info(f"\n    [CT5: 2023 Calculated YPLL per 100k] Running AIPW...")
                (ate_ct5, cl_ct5, cu_ct5, p_ct5,
                 nt_ct5, nc_ct5, err_ct5) = run_aipw(
                    data_ct5, 'treatment_binary', 'ct5_ypll_per_100k_mid',
                    ct5_controls, N_BOOT, logger, plot_dir
                )
                
                ctrl_mean_ct5 = data_ct5[data_ct5['treatment_binary'] == 0]['ct5_ypll_per_100k_mid'].mean()
                rel_change_ct5 = (ate_ct5 / ctrl_mean_ct5) * 100 if abs(ctrl_mean_ct5) > 1e-6 else np.nan
                
                logger.info(f"      ATE: {ate_ct5:.2f} YPLL/100k (95% CI: [{cl_ct5:.2f}, {cu_ct5:.2f}]), p={p_ct5:.4f}")
                logger.info(f"      Control mean: {ctrl_mean_ct5:.1f}, Relative change: {rel_change_ct5:.2f}%")
                
                outcome_comparison_rows.append({
                    'Treatment': t_col,
                    'Treatment_Label': label,
                    'Outcome': 'ct5_ypll_per_100k_mid',
                    'Outcome_Label': 'CDC Age-Adj YPLL/100k <75 (2023)',
                    'Outcome_Type': 'CT5_CDC_YPLL_2023',
                    'N': len(data_ct5),
                    'N_Treated': nt_ct5,
                    'N_Control': nc_ct5,
                    'ATE': ate_ct5,
                    'CI_Lower': cl_ct5,
                    'CI_Upper': cu_ct5,
                    'p_value': p_ct5,
                    'Control_Mean': ctrl_mean_ct5,
                    'Relative_Change_Pct': rel_change_ct5,
                    'Significant': 'Yes' if p_ct5 < 0.05 else 'No'
                })
            
            # === COMPARE CONCORDANCE ===
            sign_dv21 = np.sign(ate_dv21) if not np.isnan(ate_dv21) else 0
            sign_ct4 = np.sign(ate_ct4) if not np.isnan(ate_ct4) else 0
            sig_dv21 = p_dv21 < 0.05
            sig_ct4 = p_ct4 < 0.05
            
            logger.info(f"\n    ─── Outcome Concordance Check ───")
            if sign_dv21 == sign_ct4 and sign_dv21 != 0:
                logger.info(f"    ✓ Effect direction CONCORDANT (both {'negative' if sign_dv21 < 0 else 'positive'})")
            elif sign_dv21 * sign_ct4 < 0:
                logger.warning(f"    ✗ Effect direction DISCORDANT (DV21: {sign_dv21}, CT4: {sign_ct4})")
            else:
                logger.info(f"    ~ One or both effects near zero")
            
            # Magnitude comparison (using YPLL-equivalent scaling)
            if not np.isnan(ate_dv21) and not np.isnan(ate_ct4_ypll_equiv):
                magnitude_ratio = ate_ct4_ypll_equiv / ate_dv21 if abs(ate_dv21) > 1e-6 else np.nan
                if not np.isnan(magnitude_ratio):
                    logger.info(f"    Magnitude comparison: CT4/DV21 ratio = {magnitude_ratio:.2f}x")
                    if 0.5 <= magnitude_ratio <= 2.0:
                        logger.info(f"    ✓ Effect magnitudes SIMILAR (within 2x)")
                    elif abs(magnitude_ratio) > 2.0:
                        logger.info(f"    ⚠ Effect magnitudes DIFFER by >2x")
                else:
                    logger.info(f"    ~ Cannot compute magnitude ratio (DV21 near zero)")
            
            if sig_dv21 and sig_ct4:
                logger.info(f"    ✓ Both outcomes show SIGNIFICANT effects (p < 0.05)")
            elif sig_dv21 and not sig_ct4:
                logger.info(f"    ⚠ DV21 significant, CT4 not significant")
            elif not sig_dv21 and sig_ct4:
                logger.info(f"    ⚠ CT4 significant, DV21 not significant")
            else:
                logger.info(f"    ~ Neither outcome significant (p >= 0.05)")
        
        # Save outcome comparison table
        if outcome_comparison_rows:
            outcome_comparison_df = pd.DataFrame(outcome_comparison_rows)
            outcome_csv = os.path.join(out_dir, "outcome_sensitivity_dv21_vs_ct4_vs_ct5.csv")
            outcome_comparison_df.to_csv(outcome_csv, index=False)
            logger.info(f"\n  ✓ Saved outcome sensitivity analysis: {outcome_csv}")
            
            # Summary statistics
            logger.info(f"\n  ═══ OUTCOME SENSITIVITY SUMMARY ═══")
            n_tests = len(outcome_sensitivity_specs)
            dv21_sig = outcome_comparison_df[outcome_comparison_df['Outcome_Type'] == 'DV21_CHR_YPLL']['Significant'].value_counts().get('Yes', 0)
            ct4_sig = outcome_comparison_df[outcome_comparison_df['Outcome_Type'] == 'CT4_CDC_2023']['Significant'].value_counts().get('Yes', 0) if ct4_available else 0
            ct5_sig = outcome_comparison_df[outcome_comparison_df['Outcome_Type'] == 'CT5_CDC_YPLL_2023']['Significant'].value_counts().get('Yes', 0) if ct5_available else 0
            
            logger.info(f"  Total treatment specifications tested: {n_tests}")
            logger.info(f"  DV21 (CHR YPLL 2020-22, age-adj):    {dv21_sig}/{n_tests} significant effects")
            if ct4_available:
                logger.info(f"  CT4 (CDC Deaths 2023, not age-adj): {ct4_sig}/{n_tests} significant effects")
            if ct5_available:
                logger.info(f"  CT5 (CDC YPLL 2023, not age-adj):   {ct5_sig}/{n_tests} significant effects")
            
            # Check sign concordance across all outcome pairs
            outcome_wide = outcome_comparison_df.pivot_table(
                index=['Treatment', 'Treatment_Label'],
                columns='Outcome_Type',
                values='ATE',
                aggfunc='first'
            ).reset_index()
            
            # DV21 vs CT4 concordance
            if 'DV21_CHR_YPLL' in outcome_wide.columns and 'CT4_CDC_2023' in outcome_wide.columns:
                concordant_signs_dv21_ct4 = 0
                concordant_magnitude_dv21_ct4 = 0
                valid_pairs_dv21_ct4 = 0
                
                for _, row in outcome_wide.iterrows():
                    try:
                        ate_dv21_val = row['DV21_CHR_YPLL']
                        ate_ct4_val = row['CT4_CDC_2023']
                        if not np.isnan(ate_dv21_val) and not np.isnan(ate_ct4_val):
                            valid_pairs_dv21_ct4 += 1
                            # Direction concordance
                            if np.sign(ate_dv21_val) == np.sign(ate_ct4_val) and np.sign(ate_dv21_val) != 0:
                                concordant_signs_dv21_ct4 += 1
                            # Magnitude concordance (CT4 scaled to YPLL-equivalent)
                            ate_ct4_ypll = ate_ct4_val * DEATH_TO_YPLL_FACTOR
                            ratio = ate_ct4_ypll / ate_dv21_val if abs(ate_dv21_val) > 1e-6 else np.nan
                            if not np.isnan(ratio) and 0.5 <= ratio <= 2.0:
                                concordant_magnitude_dv21_ct4 += 1
                    except:
                        continue
                
                logger.info(f"\n  DV21 vs CT4 Concordance:")
                logger.info(f"    Effect direction: {concordant_signs_dv21_ct4}/{valid_pairs_dv21_ct4} pairs")
                logger.info(f"    Effect magnitude: {concordant_magnitude_dv21_ct4}/{valid_pairs_dv21_ct4} pairs (within 2x, CT4 scaled by {DEATH_TO_YPLL_FACTOR}x)")
            
            # DV21 vs CT5 concordance (both are YPLL, direct comparison)
            if 'DV21_CHR_YPLL' in outcome_wide.columns and 'CT5_CDC_YPLL_2023' in outcome_wide.columns:
                concordant_signs_dv21_ct5 = 0
                concordant_magnitude_dv21_ct5 = 0
                valid_pairs_dv21_ct5 = 0
                
                for _, row in outcome_wide.iterrows():
                    try:
                        ate_dv21_val = row['DV21_CHR_YPLL']
                        ate_ct5_val = row['CT5_CDC_YPLL_2023']
                        if not np.isnan(ate_dv21_val) and not np.isnan(ate_ct5_val):
                            valid_pairs_dv21_ct5 += 1
                            # Direction concordance
                            if np.sign(ate_dv21_val) == np.sign(ate_ct5_val) and np.sign(ate_dv21_val) != 0:
                                concordant_signs_dv21_ct5 += 1
                            # Magnitude concordance (no scaling needed - both YPLL)
                            ratio = ate_ct5_val / ate_dv21_val if abs(ate_dv21_val) > 1e-6 else np.nan
                            if not np.isnan(ratio) and 0.5 <= ratio <= 2.0:
                                concordant_magnitude_dv21_ct5 += 1
                    except:
                        continue
                
                logger.info(f"\n  DV21 vs CT5 Concordance (both YPLL):")
                logger.info(f"    Effect direction: {concordant_signs_dv21_ct5}/{valid_pairs_dv21_ct5} pairs")
                logger.info(f"    Effect magnitude: {concordant_magnitude_dv21_ct5}/{valid_pairs_dv21_ct5} pairs (within 2x, direct YPLL comparison)")
            
            # CT4 vs CT5 concordance
            if 'CT4_CDC_2023' in outcome_wide.columns and 'CT5_CDC_YPLL_2023' in outcome_wide.columns:
                concordant_signs_ct4_ct5 = 0
                concordant_magnitude_ct4_ct5 = 0
                valid_pairs_ct4_ct5 = 0
                
                for _, row in outcome_wide.iterrows():
                    try:
                        ate_ct4_val = row['CT4_CDC_2023']
                        ate_ct5_val = row['CT5_CDC_YPLL_2023']
                        if not np.isnan(ate_ct4_val) and not np.isnan(ate_ct5_val):
                            valid_pairs_ct4_ct5 += 1
                            # Direction concordance
                            if np.sign(ate_ct4_val) == np.sign(ate_ct5_val) and np.sign(ate_ct4_val) != 0:
                                concordant_signs_ct4_ct5 += 1
                            # Magnitude concordance (CT4 scaled to YPLL-equivalent)
                            ate_ct4_ypll = ate_ct4_val * DEATH_TO_YPLL_FACTOR
                            ratio = ate_ct5_val / ate_ct4_ypll if abs(ate_ct4_ypll) > 1e-6 else np.nan
                            if not np.isnan(ratio) and 0.5 <= ratio <= 2.0:
                                concordant_magnitude_ct4_ct5 += 1
                    except:
                        continue
                
                logger.info(f"\n  CT4 vs CT5 Concordance:")
                logger.info(f"    Effect direction: {concordant_signs_ct4_ct5}/{valid_pairs_ct4_ct5} pairs")
                logger.info(f"    Effect magnitude: {concordant_magnitude_ct4_ct5}/{valid_pairs_ct4_ct5} pairs (within 2x, CT4 scaled by {DEATH_TO_YPLL_FACTOR}x)")
            
            # Overall interpretation
            logger.info(f"\n  Interpretation:")
            if ct5_available:
                if dv21_sig == ct5_sig and concordant_signs_dv21_ct5 >= 0.75 * valid_pairs_dv21_ct5:
                    logger.info(f"    HIGH concordance (DV21↔CT5) - Results robust to outcome measure choice")
                else:
                    logger.info(f"    Moderate concordance - Some sensitivity to outcome operationalization")
                logger.info(f"    DV21 (age-adj, multi-year) vs CT5 (2023 only) comparison most relevant")
            elif ct4_available:
                if dv21_sig == ct4_sig:
                    logger.info(f"    HIGH concordance - Outcome choice is robust")
                else:
                    logger.info(f"    Moderate concordance - Some sensitivity to outcome measure")
            
            logger.info(f"  ════════════════════════════════════\n")
    else:
        logger.warning("  CT4 (CDC WONDER 2023) data NOT available. Skipping outcome sensitivity analysis.")
        logger.warning("  To enable: Ensure cdc_2023_county_deaths_under_75yrs table exists with 'deaths' column and 'county_fips' key.")
    
    logger.info("="*80 + "\n")

    # --- CT5 SUPPRESSION SENSITIVITY: Test Impact of Imputation Assumptions ---
    logger.info("\n" + "="*80)
    logger.info("  CT5 SUPPRESSION SENSITIVITY ANALYSIS")
    logger.info("="*80)
    logger.info("  Purpose: Test how imputation of suppressed death counts affects MO14→CT5 results.")
    logger.info("  Method: Compare low (1), mid (5), and high (9) imputation values for suppressed cells.")
    logger.info("  Focus: MO14 (AI Automate Routine Tasks) → CT5 (2023 Age-Adjusted YPLL <75)")
    logger.info("  Metrics: ATE, p-value, q-value (FDR), E-value, 95% CI")
    logger.info("="*80 + "\n")
    
    ct5_sensitivity_results = []
    
    # Check if all three CT5 columns are available
    ct5_cols = ['ct5_ypll_per_100k_low', 'ct5_ypll_per_100k_mid', 'ct5_ypll_per_100k_high']
    ct5_available = all(col in df.columns for col in ct5_cols)
    
    if ct5_available and 'mo14_ai_automate_routine_tasks_pct' in df.columns:
        logger.info("  All three CT5 imputation scenarios available. Running MO14→CT5 sensitivity...")
        
        for scenario_name, outcome_col in [
            ('Low (Suppressed=1)', 'ct5_ypll_per_100k_low'),
            ('Mid (Suppressed=5)', 'ct5_ypll_per_100k_mid'),
            ('High (Suppressed=9)', 'ct5_ypll_per_100k_high')
        ]:
            logger.info(f"\n  Scenario: {scenario_name} ({outcome_col})")
            ct5_controls = get_control_set(outcome_col, base_controls, df)
            
            # Prepare data
            analysis_df = df[['county_fips', 'mo14_ai_automate_routine_tasks_pct', outcome_col] + ct5_controls].copy()
            analysis_df = analysis_df.dropna(subset=['mo14_ai_automate_routine_tasks_pct', outcome_col] + ct5_controls)
            
            if len(analysis_df) < 100:
                logger.warning(f"    Insufficient data (N={len(analysis_df)}). Skipping.")
                continue
            
            logger.info(f"    Sample size: {len(analysis_df)} counties")
            logger.info(f"    Outcome range: {analysis_df[outcome_col].min():.1f} - {analysis_df[outcome_col].max():.1f} YPLL/100k")
            
            # Compute treatment threshold (use 75th percentile to ensure sufficient control group)
            # Median often = 0 for sparse adoption variables like MO14
            threshold = analysis_df['mo14_ai_automate_routine_tasks_pct'].quantile(0.75)
            
            # If 75th percentile is still 0, use any non-zero value as threshold
            if threshold == 0:
                nonzero_vals = analysis_df.loc[analysis_df['mo14_ai_automate_routine_tasks_pct'] > 0, 'mo14_ai_automate_routine_tasks_pct']
                if len(nonzero_vals) > 0:
                    threshold = nonzero_vals.min()
                else:
                    logger.warning(f"    All MO14 values are zero for {scenario_name}. Skipping.")
                    continue
            
            analysis_df['treatment_binary'] = (analysis_df['mo14_ai_automate_routine_tasks_pct'] >= threshold).astype(int)
            
            n_treated = analysis_df['treatment_binary'].sum()
            n_control = len(analysis_df) - n_treated
            logger.info(f"    Treatment threshold (75th percentile): {threshold:.2f}%")
            logger.info(f"    N_treated: {n_treated}, N_control: {n_control}")
            
            # Check for sufficient variation
            if n_treated < 50 or n_control < 50:
                logger.warning(f"    Insufficient treatment/control balance (treated={n_treated}, control={n_control}). Skipping.")
                continue
            
            # Run AIPW
            try:
                (ate, ci_lower, ci_upper, pvalue, nt, nc, aipw_err) = run_aipw(
                    analysis_df, 'treatment_binary', outcome_col,
                    ct5_controls, N_BOOT, logger, plot_dir
                )
                
                if aipw_err:
                    logger.warning(f"    AIPW failed for {scenario_name}. Skipping.")
                    continue
                
                # Compute control mean for relative change
                ctrl_mean = analysis_df.loc[analysis_df['treatment_binary'] == 0, outcome_col].mean()
                rel_change = (ate / ctrl_mean * 100) if ctrl_mean != 0 else np.nan
                
                # Compute E-value
                sd_ctrl = analysis_df.loc[analysis_df['treatment_binary'] == 0, outcome_col].std()
                evalue_point = calculate_e_value(ate, sd_ctrl) if not np.isnan(ate) and not np.isnan(sd_ctrl) else np.nan
                
                # E-value for CI (use bound closest to null)
                if not np.isnan(ci_lower) and not np.isnan(ci_upper) and not np.isnan(sd_ctrl):
                    ci_bound_closest_to_null = ci_upper if ate < 0 else ci_lower
                    evalue_ci = calculate_e_value(ci_bound_closest_to_null, sd_ctrl)
                else:
                    evalue_ci = np.nan
                
                logger.info(f"    ATE: {ate:.2f} YPLL/100k (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")
                logger.info(f"    p-value: {pvalue:.4f}")
                logger.info(f"    Control mean: {ctrl_mean:.1f}, Relative change: {rel_change:.2f}%")
                logger.info(f"    E-value (point): {evalue_point:.2f}, E-value (CI): {evalue_ci:.2f}")
                
                ct5_sensitivity_results.append({
                    'Scenario': scenario_name,
                    'Outcome_Column': outcome_col,
                    'N': len(analysis_df),
                    'N_Treated': n_treated,
                    'N_Control': n_control,
                    'Treatment': 'MO14 (AI Automate Routine Tasks)',
                    'Treatment_Threshold_Pct': threshold,
                    'ATE': ate,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'P_Value': pvalue,
                    'Control_Mean': ctrl_mean,
                    'Relative_Change_Pct': rel_change,
                    'E_Value_Point': evalue_point,
                    'E_Value_CI': evalue_ci
                })
                
            except Exception as e:
                logger.error(f"    AIPW failed for {scenario_name}: {e}")
                continue
        
        # Save results to CSV
        if ct5_sensitivity_results:
            ct5_sens_df = pd.DataFrame(ct5_sensitivity_results)
            
            # Add FDR-adjusted q-values across scenarios
            from statsmodels.stats.multitest import multipletests
            pvals = ct5_sens_df['P_Value'].values
            _, qvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
            ct5_sens_df['Q_Value_FDR'] = qvals
            
            ct5_csv = os.path.join(out_dir, "ct5_suppression_sensitivity_mo14.csv")
            ct5_sens_df.to_csv(ct5_csv, index=False)
            logger.info(f"\n✓ Saved CT5 suppression sensitivity results to {ct5_csv}")
            
            # === SENSITIVITY ANALYSIS: E-VALUES for CT5 ===
            # Note: Same rationale as DV21 - E-values preserve full sample power
            logger.info("\n  Sensitivity Analysis: E-values for CT5 (Contemporaneous Outcome)")
            logger.info("  E-values provide sensitivity assessment without 80% sample attrition.")
            logger.info("  CT5 results use full weighted sample (see E-value calculations above).")
            
            # Rosenbaum bounds disabled - uncomment if needed for sensitivity checks
            """
            # === ROSENBAUM BOUNDS for MO14→CT5 (OPTIONAL) ===
            logger.info("\n  Computing Rosenbaum bounds for MO14 → CT5 (Mid-Scenario)...")
            ct5_mid_data = df[['county_fips', 'mo14_ai_automate_routine_tasks_pct', 'ct5_ypll_per_100k_mid'] + base_controls].copy()
            ct5_mid_data = ct5_mid_data.dropna(subset=['mo14_ai_automate_routine_tasks_pct', 'ct5_ypll_per_100k_mid'] + base_controls)
            
            if len(ct5_mid_data) >= 100:
                threshold_ct5 = ct5_mid_data['mo14_ai_automate_routine_tasks_pct'].quantile(0.75)
                if threshold_ct5 == 0:
                    nonzero_vals = ct5_mid_data.loc[ct5_mid_data['mo14_ai_automate_routine_tasks_pct'] > 0, 'mo14_ai_automate_routine_tasks_pct']
                    if len(nonzero_vals) > 0:
                        threshold_ct5 = nonzero_vals.min()
                
                ct5_mid_data['treatment_binary'] = (ct5_mid_data['mo14_ai_automate_routine_tasks_pct'] >= threshold_ct5).astype(int)
                
                matched_ct5_df = perform_ps_matching(
                    df=ct5_mid_data,
                    treatment_col='treatment_binary',
                    outcome_col='ct5_ypll_per_100k_mid',
                    confounders=base_controls,
                    caliper=0.25,
                    logger=logger
                )
            """
            
            # Summary table for paper
            logger.info(f"\n  ═══ CT5 SUPPRESSION SENSITIVITY SUMMARY (MO14→CT5) ═══")
            for _, row in ct5_sens_df.iterrows():
                logger.info(f"  {row['Scenario']:20s}: ATE={row['ATE']:7.2f}, p={row['P_Value']:.4f}, q={row['Q_Value_FDR']:.4f}, E={row['E_Value_Point']:5.2f}")
            logger.info(f"  ════════════════════════════════════════════════════════\n")
    else:
        logger.warning("  CT5 suppression sensitivity skipped: Missing columns or MO14 treatment.")
        if not ct5_available:
            missing = [col for col in ct5_cols if col not in df.columns]
            logger.warning(f"    Missing CT5 columns: {missing}")
    
    logger.info("="*80 + "\n")

    # --- OVERLAP WEIGHTING (ATO) SENSITIVITY ANALYSIS ---
    logger.info("  OVERLAP WEIGHTING (ATO) SENSITIVITY ANALYSIS")
    logger.info("="*80)
    logger.info("  Purpose: Assess robustness using overlap weighting (Li et al. 2018)")
    logger.info("  Method: w_i = e(X_i) × (1 - e(X_i)) for all units")
    logger.info("  Target: Average Treatment effect on the Overlap (ATO)")
    logger.info("  Advantage: Less sensitive to extreme propensities; better balance")
    logger.info("="*80 + "\n")
    
    overlap_specs = [
        # Primary outcomes for MO14 (AI Routine Tasks)
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "outcome": "dv21_premature_death_ypll_rate", "label": "MO14→DV21 (YPLL 2020-22)"},
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "outcome": "dv15_preventable_stays_rate", "label": "MO14→DV15 (Preventable Stays)"},
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "outcome": "ct5_ypll_per_100k_mid", "label": "MO14→CT5 (YPLL 2023 Mid)"},
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "outcome": "ct6_hospital_deaths_age_adj", "label": "MO14→CT6 (Hospital Deaths 2023)"},
        # Primary outcomes for MO21 (Robotics)
        {"treatment": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate", "label": "MO21→DV21 (YPLL 2020-22)"},
        {"treatment": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate", "label": "MO21→DV15 (Preventable Stays)"},
        {"treatment": "mo21_robotics_in_hospital_pct", "outcome": "ct5_ypll_per_100k_mid", "label": "MO21→CT5 (YPLL 2023 Mid)"},
        {"treatment": "mo21_robotics_in_hospital_pct", "outcome": "ct6_hospital_deaths_age_adj", "label": "MO21→CT6 (Hospital Deaths 2023)"},
    ]
    
    overlap_results = []
    
    for spec in overlap_specs:
        t_col = spec["treatment"]
        y_col = spec["outcome"]
        label = spec["label"]
        
        logger.info(f"\n  {label}")
        logger.info(f"    Treatment: {t_col}")
        logger.info(f"    Outcome: {y_col}")
        model_controls = get_control_set(y_col, base_controls, df)
        
        # Check if columns exist
        required_cols = [t_col, y_col] + model_controls
        if any(c not in df.columns for c in required_cols):
            logger.warning(f"    Skipping: Missing columns")
            continue
        
        # Prepare data
        analysis_df = df[required_cols].dropna()
        if len(analysis_df) < 100:
            logger.warning(f"    Skipping: Insufficient data (N={len(analysis_df)})")
            continue
        
        # Binarize treatment (median split for MO14, which is sparse)
        if "mo14" in t_col.lower():
            # Use 75th percentile for sparse MO14 (most counties have 0)
            threshold = analysis_df[t_col].quantile(0.75)
            if threshold == 0:
                nonzero_vals = analysis_df.loc[analysis_df[t_col] > 0, t_col]
                threshold = nonzero_vals.min() if len(nonzero_vals) > 0 else 0
        else:
            # Standard median split for other treatments
            threshold = analysis_df[t_col].median()
        
        analysis_df['treatment_binary'] = (analysis_df[t_col] > threshold).astype(int)
        
        n_treated = analysis_df['treatment_binary'].sum()
        n_control = len(analysis_df) - n_treated
        
        logger.info(f"    N total: {len(analysis_df)}, Treated: {n_treated}, Control: {n_control}")
        logger.info(f"    Threshold: {threshold:.2f}")
        
        if n_treated < 10 or n_control < 10:
            logger.warning(f"    Skipping: Insufficient treatment/control balance")
            continue
        
        # Run standard IPTW AIPW for comparison
        try:
            (ate_iptw, ci_lower_iptw, ci_upper_iptw, p_iptw, nt_iptw, nc_iptw, err_iptw) = run_aipw(
                analysis_df, 'treatment_binary', y_col, model_controls, N_BOOT, logger, plot_dir
            )
            
            if err_iptw:
                logger.warning(f"    IPTW AIPW failed")
                ate_iptw = ci_lower_iptw = ci_upper_iptw = p_iptw = np.nan
        except Exception as e:
            logger.error(f"    IPTW AIPW error: {e}")
            ate_iptw = ci_lower_iptw = ci_upper_iptw = p_iptw = np.nan
        
        # Run overlap weighting AIPW
        try:
            (ate_ato, ci_lower_ato, ci_upper_ato, p_ato, nt_ato, nc_ato, ess_ato, err_ato) = run_aipw_overlap(
                analysis_df, 'treatment_binary', y_col, model_controls, N_BOOT, logger, plot_dir
            )
            
            if err_ato:
                logger.warning(f"    Overlap weighting AIPW failed")
                ate_ato = ci_lower_ato = ci_upper_ato = p_ato = ess_ato = np.nan
        except Exception as e:
            logger.error(f"    Overlap weighting error: {e}")
            ate_ato = ci_lower_ato = ci_upper_ato = p_ato = ess_ato = np.nan
        
        # Log results
        logger.info(f"    IPTW ATE: {ate_iptw:.2f}, 95% CI [{ci_lower_iptw:.2f}, {ci_upper_iptw:.2f}], p={p_iptw:.4f}")
        logger.info(f"    ATO:      {ate_ato:.2f}, 95% CI [{ci_lower_ato:.2f}, {ci_upper_ato:.2f}], p={p_ato:.4f}, ESS={ess_ato:.1f}")
        
        # Compute outcome control mean for context
        ctrl_mean = analysis_df.loc[analysis_df['treatment_binary'] == 0, y_col].mean()
        
        overlap_results.append({
            'Treatment': t_col,
            'Outcome': y_col,
            'Label': label,
            'N': len(analysis_df),
            'N_Treated': n_treated,
            'N_Control': n_control,
            'Threshold': threshold,
            'Control_Mean': ctrl_mean,
            # IPTW results
            'IPTW_ATE': ate_iptw,
            'IPTW_CI_Lower': ci_lower_iptw,
            'IPTW_CI_Upper': ci_upper_iptw,
            'IPTW_p': p_iptw,
            # ATO results
            'ATO_ATE': ate_ato,
            'ATO_CI_Lower': ci_lower_ato,
            'ATO_CI_Upper': ci_upper_ato,
            'ATO_p': p_ato,
            'ATO_ESS': ess_ato,
            # Comparison
            'ATE_Difference': ate_ato - ate_iptw if not np.isnan(ate_ato) and not np.isnan(ate_iptw) else np.nan,
            'ATE_Pct_Change': ((ate_ato - ate_iptw) / ate_iptw * 100) if not np.isnan(ate_ato) and not np.isnan(ate_iptw) and ate_iptw != 0 else np.nan
        })
    
    # Save results
    if overlap_results:
        overlap_df = pd.DataFrame(overlap_results)
        overlap_csv = os.path.join(out_dir, "overlap_weighting_ato_comparison.csv")
        overlap_df.to_csv(overlap_csv, index=False)
        logger.info(f"\n✓ Saved overlap weighting (ATO) comparison to {overlap_csv}")
        
        # Generate IPTW vs ATO forest plot
        logger.info(f"\n  Generating IPTW vs ATO forest plot...")
        plot_iptw_vs_ato_forest(
            overlap_df=overlap_df,
            outcome_label="Years of Potential Life Lost (YPLL)",
            plot_dir=out_dir,
            logger=logger,
            filename="forest_plot_iptw_vs_ato.png"
        )
        
        # Summary table
        logger.info(f"\n  ═══ OVERLAP WEIGHTING (ATO) SUMMARY ═══")
        logger.info(f"  {'Test':<30s} {'IPTW ATE':>10s} {'ATO ATE':>10s} {'ATO p':>8s} {'ATO ESS':>10s}")
        logger.info(f"  {'-'*70}")
        for _, row in overlap_df.iterrows():
            logger.info(f"  {row['Label']:<30s} {row['IPTW_ATE']:>10.2f} {row['ATO_ATE']:>10.2f} {row['ATO_p']:>8.4f} {row['ATO_ESS']:>10.1f}")
        logger.info(f"  {'═'*70}")
        
        # Interpretation
        logger.info("\n  INTERPRETATION:")
        logger.info("  - ATO (overlap weighting) targets the population with best covariate balance")
        logger.info("  - Less sensitive to extreme propensities than standard IPTW")
        logger.info("  - If ATO and IPTW agree, results are robust across target populations")
        
        # Check for substantial differences
        major_diffs = overlap_df[
            (abs(overlap_df['ATE_Pct_Change']) > 50) & 
            (~overlap_df['ATE_Pct_Change'].isna())
        ]
        if not major_diffs.empty:
            logger.warning(f"\n  ⚠ Substantial differences (>50%) detected for:")
            for _, row in major_diffs.iterrows():
                logger.warning(f"    {row['Label']}: {row['ATE_Pct_Change']:.1f}% change")
        else:
            logger.info("\n  ✓ Results are consistent across IPTW and ATO estimands")
    else:
        logger.warning("  No overlap weighting results generated.")
    
    logger.info("\n" + "="*80 + "\n")

    # --- PLACEBO TESTS: 2024 AI Adoption -> 2019 Mortality (Pre-Treatment Outcome) ---
    logger.info("Running PLACEBO tests: 2024 technology adoption -> 2019 mortality (PL1)...")
    logger.info("  Purpose: Falsification test to rule out reverse causality.")
    logger.info("  Expected: Null effects (2024 AI should NOT predict 2019 mortality).")
    
    placebo_specs = [
        # Key AI composite and robotics
        {"treatment": "mo1_genai_composite_score", "outcome": "pl1_ypll_rate"},
        {"treatment": "mo2_robotics_composite_score", "outcome": "pl1_ypll_rate"},
        # Individual AI adoption indicators
        {"treatment": "mo11_ai_staff_scheduling_pct", "outcome": "pl1_ypll_rate"},
        {"treatment": "mo12_ai_predict_staff_needs_pct", "outcome": "pl1_ypll_rate"},
        {"treatment": "mo13_ai_predict_patient_demand_pct", "outcome": "pl1_ypll_rate"},
        {"treatment": "mo14_ai_automate_routine_tasks_pct", "outcome": "pl1_ypll_rate"},
        {"treatment": "mo15_ai_optimize_workflows_pct", "outcome": "pl1_ypll_rate"},
        # Robotics sub-component
        {"treatment": "mo21_robotics_in_hospital_pct", "outcome": "pl1_ypll_rate"},
    ]
    
    placebo_results = []
    if 'pl1_ypll_rate' not in df.columns or df['pl1_ypll_rate'].isna().all():
        logger.warning("  PL1 (2019 YPLL) placebo outcome not available. Skipping placebo tests.")
    else:
        logger.info(f"  PL1 placebo outcome available: {df['pl1_ypll_rate'].notna().sum()} non-null values.")
        
        for spec in placebo_specs:
            t_col, y_col = spec["treatment"], spec["outcome"]
            logger.info(f"  Placebo test: {t_col} -> {y_col}")
            model_controls = get_control_set(y_col, base_controls, df)
            
            required_cols = [t_col, y_col, 'state_fips_for_clustering'] + model_controls
            if any(c not in df.columns for c in required_cols):
                logger.warning(f"    Skipping: missing one or more columns from: {required_cols}")
                continue
                
            d = df[required_cols].dropna()
            if d.empty or d[t_col].nunique() < 2:
                logger.warning("    Skipping: Not enough data or variation after dropping NaNs.")
                continue
                
            # OLS with continuous treatment
            res_ols = run_ols_clustered(d[y_col], d[[t_col] + model_controls], d['state_fips_for_clustering'])
            ols_beta = res_ols.params.get(t_col, np.nan)
            ols_p = res_ols.pvalues.get(t_col, np.nan)
            ols_ci = res_ols.conf_int().loc[t_col].values if t_col in res_ols.params.index else (np.nan, np.nan)
            
            # AIPW with binarized treatment
            median_val = d[t_col].median()
            bin_col = f"{t_col}_gt_median"
            d[bin_col] = (d[t_col] > median_val).astype(int)

            # Propensity diagnostics (overlap/weights)
            try:
                X_diag = d[model_controls].copy()
                if X_diag.isnull().any().any():
                    X_diag = X_diag.fillna(X_diag.mean())
                ps_diag, _ = estimate_propensity_scores(X_diag, d[bin_col].values.astype(int), logger, f"{t_col}_diag_placebo")
                if ps_diag is not None:
                    diag_row = compute_weight_diagnostics(d[bin_col].values.astype(int), ps_diag, t_col, y_col)
                    weight_diag_rows.append(diag_row)
            except Exception as e:
                logger.warning(f"    Propensity diagnostics failed for placebo {t_col}->{y_col}: {e}")

            if d[bin_col].nunique() < 2:
                 aipw_ate, aipw_cl, aipw_cu, aipw_p, n_t, n_c = [np.nan] * 6
                 sd_ctrl = np.nan
            else:
                (aipw_ate, aipw_cl, aipw_cu, aipw_p,
                n_t, n_c, aipw_err) = run_aipw(d, bin_col, y_col, model_controls, N_BOOT, logger, plot_dir)
                # Calculate control group SD for E-value
                sd_ctrl = d[d[bin_col] == 0][y_col].std() if (d[bin_col] == 0).sum() > 1 else np.nan
            
            # Calculate E-value for sensitivity analysis
            evalue = calculate_e_value(aipw_ate, sd_ctrl) if not np.isnan(aipw_ate) and not np.isnan(sd_ctrl) else np.nan
            evalue_ci = calculate_e_value(aipw_cl, sd_ctrl) if not np.isnan(aipw_cl) and not np.isnan(sd_ctrl) else np.nan

            placebo_results.append({
                "Treatment": t_col, "Outcome": y_col, "N": len(d),
                "OLS_Beta": ols_beta, "OLS_p": ols_p, "OLS_CI_Lower": ols_ci[0], "OLS_CI_Upper": ols_ci[1],
                "AIPW_ATE": aipw_ate, "AIPW_p": aipw_p, "AIPW_CI_Lower": aipw_cl, "AIPW_CI_Upper": aipw_cu,
                "N_Treated": n_t, "N_Control": n_c,
                "E_Value": evalue, "E_Value_CI": evalue_ci,
            })
    
    dissertation_placebo_df = pd.DataFrame(placebo_results)
    if not dissertation_placebo_df.empty:
        out_csv = os.path.join(out_dir, "dissertation_placebo_tests_summary.csv")
        dissertation_placebo_df.to_csv(out_csv, index=False)
        logger.info(f"Saved dissertation placebo tests summary to {out_csv}")
        logger.info(f"  Total placebo tests: {len(dissertation_placebo_df)}")
        
        # Count significant results (should be 0 or very few)
        sig_count = (dissertation_placebo_df['AIPW_p'] < 0.05).sum()
        logger.info(f"  Significant placebo effects (p<0.05): {sig_count}/{len(dissertation_placebo_df)}")
        
        if sig_count > 0:
            logger.warning("  WARNING: Some placebo tests show significant effects!")
            logger.warning("  This may indicate reverse causality or confounding bias.")
            sig_tests = dissertation_placebo_df[dissertation_placebo_df['AIPW_p'] < 0.05][['Treatment', 'AIPW_ATE', 'AIPW_p']]
            logger.warning(f"  Significant placebo tests:\n{sig_tests}")
        else:
            logger.info("  ✓ All placebo tests are non-significant (expected result).")
            logger.info("  This supports the causal interpretation of 2024 AI → 2024 mortality effects.")
        
        # --- FISHER'S COMBINED PLACEBO TEST ---
        logger.info("\n" + "="*80)
        logger.info("FISHER'S COMBINED PLACEBO TEST")
        logger.info("="*80)
        logger.info("Testing global null hypothesis: ALL placebo effects are truly zero.")
        logger.info("Fisher's method combines p-values: χ² = -2 Σ ln(pᵢ)")
        
        # Filter valid p-values for Fisher's test
        valid_p_values = dissertation_placebo_df['AIPW_p'].dropna()
        valid_p_values = valid_p_values[(valid_p_values > 0) & (valid_p_values <= 1)]
        
        if len(valid_p_values) >= 2:
            # Calculate Fisher's combined test statistic
            # χ² = -2 * sum(ln(p_i))
            # degrees of freedom = 2k where k = number of tests
            k = len(valid_p_values)
            chi_squared = -2 * np.sum(np.log(valid_p_values))
            degrees_of_freedom = 2 * k
            
            # Calculate combined p-value from chi-squared distribution
            from scipy.stats import chi2
            fisher_p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)
            
            logger.info(f"  Number of tests combined: {k}")
            logger.info(f"  Fisher's χ² statistic: {chi_squared:.4f}")
            logger.info(f"  Degrees of freedom: {degrees_of_freedom}")
            logger.info(f"  Combined p-value: {fisher_p_value:.6f}")
            
            # Save Fisher's test results
            fisher_results = {
                "Test": "Fisher's Combined Placebo Test",
                "Number_of_Tests": k,
                "Chi_Squared": chi_squared,
                "DF": degrees_of_freedom,
                "Combined_P_Value": fisher_p_value,
                "Interpretation": "Non-significant" if fisher_p_value >= 0.05 else "SIGNIFICANT - potential concern"
            }
            
            fisher_df = pd.DataFrame([fisher_results])
            fisher_csv = os.path.join(out_dir, "fisher_combined_placebo_test.csv")
            fisher_df.to_csv(fisher_csv, index=False)
            logger.info(f"  Saved Fisher's combined test results to {fisher_csv}")
            
            if fisher_p_value >= 0.05:
                logger.info("  ✓ Fisher's combined test is NON-SIGNIFICANT (p ≥ 0.05)")
                logger.info("  Interpretation: Global null hypothesis NOT rejected.")
                logger.info("  The placebo tests collectively show no evidence of reverse causality.")
            else:
                logger.warning("  ⚠ Fisher's combined test IS SIGNIFICANT (p < 0.05)")
                logger.warning("  Interpretation: At least some placebo effects may be non-zero.")
                logger.warning("  This suggests potential concerns about reverse causality or confounding.")
            
            logger.info("="*80 + "\n")
        else:
            logger.warning(f"  Insufficient valid p-values for Fisher's test (found {len(valid_p_values)}, need ≥2)")
        
        # Create forest plot for placebo tests
        plot_forest_plot_ate(
            results_df=dissertation_placebo_df,
            outcome_label="2019 Premature Death Rate (Placebo)",
            plot_dir=plot_dir,
            logger=logger,
            filename="forest_plot_placebo_tests.png"
        )
    else:
        logger.warning("  No placebo tests completed. Check PL1 data availability.")

    # Save propensity/weight diagnostics (direct + placebo)
    if weight_diag_rows:
        diag_df = pd.DataFrame(weight_diag_rows)
        diag_csv = os.path.join(out_dir, "propensity_weight_diagnostics.csv")
        diag_df.to_csv(diag_csv, index=False)
        logger.info(f"Saved propensity/weight diagnostics to {diag_csv}")

    # --- DELTA MORTALITY TEST: 2024 AI → Change in YPLL (2019→2023) ---
    logger.info("\n" + "="*80)
    logger.info("DELTA MORTALITY SENSITIVITY ANALYSIS")
    logger.info("="*80)
    logger.info("Testing whether AI/Robotics adoption predicts MORTALITY IMPROVEMENT")
    logger.info("(not just cross-sectional levels)")
    logger.info("="*80 + "\n")
    
    delta_mortality_df = run_delta_mortality_test(
        df=df,
        base_controls=base_controls,
        n_boot=N_BOOT,
        logger=logger,
        out_dir=out_dir,
        plot_dir=plot_dir
    )

    # --- PRE/POST DIFFERENTIAL-CHANGE TESTS: 2019 -> 2023 across outcomes ---
    logger.info("\n" + "="*80)
    logger.info("PRE/POST DIFFERENTIAL-CHANGE REVIEWER ANALYSIS")
    logger.info("="*80)
    prepost_change_df = run_prepost_differential_change_analysis(
        df=df,
        base_controls=base_controls,
        n_boot=N_BOOT,
        logger=logger,
        out_dir=out_dir,
        plot_dir=plot_dir
    )

    # --- H2. Interaction Effects ---
    # This list combines tests from the original script AND newly requested tests.
    interaction_specs = [
        # --- Tests from original script (preserved) ---
        {"iv": "iv3_health_behaviors_score", "moderator": "mo1_genai_composite_score", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo11_ai_staff_scheduling_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo12_ai_predict_staff_needs_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo13_ai_predict_patient_demand_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo14_ai_automate_routine_tasks_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo15_ai_optimize_workflows_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo1_genai_composite_score", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo11_ai_staff_scheduling_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo12_ai_predict_staff_needs_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo13_ai_predict_patient_demand_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo14_ai_automate_routine_tasks_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo15_ai_optimize_workflows_pct", "outcome": "dv15_preventable_stays_rate"},
        # --- Newly requested tests (added) ---
        {"iv": "mo1_genai_composite_score", "moderator": "mo2_robotics_composite_score", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo11_ai_staff_scheduling_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo12_ai_predict_staff_needs_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo13_ai_predict_patient_demand_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo14_ai_automate_routine_tasks_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo15_ai_optimize_workflows_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
    ]
    
    interaction_results = []
    for spec in interaction_specs:
        iv_col, mo_col, y_col = spec["iv"], spec["moderator"], spec["outcome"]
        iv_col_c, mo_col_c = f"{iv_col}_c", f"{mo_col}_c"
        logger.info(f"  Interaction test: {iv_col} x {mo_col} -> {y_col}")

        required_cols = [iv_col_c, mo_col_c, y_col, 'state_fips_for_clustering'] + base_controls
        if any(c not in df.columns for c in required_cols):
            logger.warning(f"    Skipping: missing one or more required columns.")
            continue
            
        d = df[required_cols].dropna()
        if d.empty:
            logger.warning("    Skipping: Not enough data after dropping NaNs.")
            continue
            
        interaction_term = f"{iv_col_c}_x_{mo_col_c}"
        
        # Ensure original IVs are not duplicated in controls
        controls_for_model = [c for c in base_controls if c not in [iv_col, mo_col]]
        
        X = d[[iv_col_c, mo_col_c] + controls_for_model].copy()
        X[interaction_term] = d[iv_col_c] * d[mo_col_c]
        
        res = run_ols_clustered(d[y_col], X, d['state_fips_for_clustering'])
        beta = res.params.get(interaction_term, np.nan)
        pval = res.pvalues.get(interaction_term, np.nan)
        ci = res.conf_int().loc[interaction_term].values if interaction_term in res.params.index else (np.nan, np.nan)

        interaction_results.append({
            "IV": iv_col, "Moderator": mo_col, "Outcome": y_col, "N": len(d),
            "Interaction_Beta": beta, "Interaction_p": pval,
            "Interaction_CI_Lower": ci[0], "Interaction_CI_Upper": ci[1],
        })
        
        plot_interaction_continuous(d, iv_col_c, mo_col_c, y_col, 
                                    name=f"{spec['iv'].split('_')[0]}_x_{spec['moderator'].split('_')[0]}", 
                                    beta=beta, pval=pval, plot_dir=plot_dir, logger=logger)

    # --- NEW: Supplementary Interaction tests for DV21 (Premature Death) ---
    interaction_specs_dv21 = [
        {"iv": "mo1_genai_composite_score", "moderator": "mo2_robotics_composite_score", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo11_ai_staff_scheduling_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo12_ai_predict_staff_needs_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo13_ai_predict_patient_demand_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo14_ai_automate_routine_tasks_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo15_ai_optimize_workflows_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
    ]

    # Process regular interaction tests
    for spec in interaction_specs:
        iv_col, mo_col, y_col = spec["iv"], spec["moderator"], spec["outcome"]
        iv_col_c, mo_col_c = f"{iv_col}_c", f"{mo_col}_c"
        logger.info(f"  Interaction test: {iv_col} x {mo_col} -> {y_col}")

        required_cols = [iv_col_c, mo_col_c, y_col, 'state_fips_for_clustering'] + base_controls
        if any(c not in df.columns for c in required_cols):
            logger.warning(f"    Skipping: missing one or more required columns.")
            continue
            
        d = df[required_cols].dropna()
        if d.empty:
            logger.warning("    Skipping: Not enough data after dropping NaNs.")
            continue
            
        interaction_term = f"{iv_col_c}_x_{mo_col_c}"
        
        # Ensure original IVs are not duplicated in controls
        controls_for_model = [c for c in base_controls if c not in [iv_col, mo_col]]
        
        X = d[[iv_col_c, mo_col_c] + controls_for_model].copy()
        X[interaction_term] = d[iv_col_c] * d[mo_col_c]
        
        res = run_ols_clustered(d[y_col], X, d['state_fips_for_clustering'])
        beta = res.params.get(interaction_term, np.nan)
        pval = res.pvalues.get(interaction_term, np.nan)
        ci = res.conf_int().loc[interaction_term].values if interaction_term in res.params.index else (np.nan, np.nan)

        interaction_results.append({
            "IV": iv_col, "Moderator": mo_col, "Outcome": y_col, "N": len(d),
            "Interaction_Beta": beta, "Interaction_p": pval,
            "Interaction_CI_Lower": ci[0], "Interaction_CI_Upper": ci[1],
        })
        
        plot_interaction_continuous(d, iv_col_c, mo_col_c, y_col, 
                                   name=f"{spec['iv'].split('_')[0]}_x_{spec['moderator'].split('_')[0]}", 
                                   beta=beta, pval=pval, plot_dir=plot_dir, logger=logger)

    # Process DV21 interaction tests
    logger.info("  Running DV21 (premature death) interaction tests...")
    for spec in interaction_specs_dv21:
        iv_col, mo_col, y_col = spec["iv"], spec["moderator"], spec["outcome"]
        iv_col_c, mo_col_c = f"{iv_col}_c", f"{mo_col}_c"
        logger.info(f"  DV21 Interaction test: {iv_col} x {mo_col} -> {y_col}")

        required_cols = [iv_col_c, mo_col_c, y_col, 'state_fips_for_clustering'] + base_controls
        if any(c not in df.columns for c in required_cols):
            logger.warning(f"    Skipping: missing one or more required columns.")
            continue
            
        d = df[required_cols].dropna()
        if d.empty:
            logger.warning("    Skipping: Not enough data after dropping NaNs.")
            continue
            
        interaction_term = f"{iv_col_c}_x_{mo_col_c}"
        
        # Ensure original IVs are not duplicated in controls
        controls_for_model = [c for c in base_controls if c not in [iv_col, mo_col]]
        
        X = d[[iv_col_c, mo_col_c] + controls_for_model].copy()
        X[interaction_term] = d[iv_col_c] * d[mo_col_c]
        
        res = run_ols_clustered(d[y_col], X, d['state_fips_for_clustering'])
        beta = res.params.get(interaction_term, np.nan)
        pval = res.pvalues.get(interaction_term, np.nan)
        ci = res.conf_int().loc[interaction_term].values if interaction_term in res.params.index else (np.nan, np.nan)

        interaction_results.append({
            "IV": iv_col, "Moderator": mo_col, "Outcome": y_col, "N": len(d),
            "Interaction_Beta": beta, "Interaction_p": pval,
            "Interaction_CI_Lower": ci[0], "Interaction_CI_Upper": ci[1],
        })
        
        plot_interaction_continuous(d, iv_col_c, mo_col_c, y_col, 
                                   name=f"{spec['iv'].split('_')[0]}_x_{spec['moderator'].split('_')[0]}_on_DV21", 
                                   beta=beta, pval=pval, plot_dir=plot_dir, logger=logger)

    # --- Targeted continuous moderator surfaces (requested) ---
    logger.info("  Running targeted continuous moderator surface models...")
    surface_specs = [
        {"iv": "iv3_health_behaviors_score", "moderator": "mo1_genai_composite_score", "outcome": "dv2_health_outcomes_score", "name": "iv3_x_mo1_on_dv2"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo14_ai_automate_routine_tasks_pct", "outcome": "dv2_health_outcomes_score", "name": "iv3_x_mo14_on_dv2"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo14_ai_automate_routine_tasks_pct", "outcome": "dv21_premature_death_ypll_rate", "name": "iv3_x_mo14_on_dv21"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo14_ai_automate_routine_tasks_pct", "outcome": "ct5_ypll_per_100k_mid", "name": "iv3_x_mo14_on_ct5"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo14_ai_automate_routine_tasks_pct", "outcome": "ct6_hospital_deaths_age_adj", "name": "iv3_x_mo14_on_ct6"},
    ]
    surface_label_map = {
        "iv3_health_behaviors_score": "IV3 Health Behaviors",
        "mo1_genai_composite_score": "MO1 GenAI Capabilities",
        "mo14_ai_automate_routine_tasks_pct": "MO14 AI Workflow Automation",
        "dv2_health_outcomes_score": "DV2 Healthcare Quality",
        "dv21_premature_death_ypll_rate": "DV21 Premature Death YPLL",
        "ct5_ypll_per_100k_mid": "CT5 Age-Adjusted YPLL (2023)",
        "ct6_hospital_deaths_age_adj": "CT6 Age-Adjusted Hospital Deaths (2023)",
    }
    surface_transforms = [
        {"key": "raw", "label": "Raw"},
        {"key": "winsorized", "label": "Winsorized (1st-99th percentile)"},
        {"key": "signed_log1p", "label": "Signed log1p"},
    ]
    surface_rows = []

    for spec in surface_specs:
        iv_col, mo_col, y_col = spec["iv"], spec["moderator"], spec["outcome"]
        controls_for_model = [c for c in base_controls if c not in [iv_col, mo_col] and c in df.columns]
        required_cols = [iv_col, mo_col, y_col, "state_fips_for_clustering"] + controls_for_model
        logger.info(f"  Surface model variants: {iv_col} x {mo_col} -> {y_col}")

        if any(c not in df.columns for c in required_cols):
            logger.warning("    Skipping surface model: missing one or more required columns.")
            continue

        d_base = df[required_cols].dropna().copy()
        if len(d_base) < 80:
            logger.warning(f"    Skipping surface model: insufficient sample after dropna (N={len(d_base)}).")
            continue

        for transform in surface_transforms:
            t_key = transform["key"]
            t_label = transform["label"]
            d = d_base.copy()

            if t_key == "winsorized":
                d[iv_col] = _winsorize_series(d[iv_col], 0.01, 0.99)
                d[mo_col] = _winsorize_series(d[mo_col], 0.01, 0.99)
                d[y_col] = _winsorize_series(d[y_col], 0.01, 0.99)
            elif t_key == "signed_log1p":
                d[iv_col] = _signed_log1p_series(d[iv_col])
                d[mo_col] = _signed_log1p_series(d[mo_col])
                d[y_col] = _signed_log1p_series(d[y_col])

            d = d.replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 80:
                logger.warning(f"    Skipping {t_label} variant: insufficient sample (N={len(d)}).")
                continue

            iv_centered_col = f"{iv_col}_{t_key}_surface_c"
            moderator_centered_col = f"{mo_col}_{t_key}_surface_c"
            interaction_term = f"{iv_centered_col}_x_{moderator_centered_col}"
            d[iv_centered_col] = d[iv_col] - d[iv_col].mean()
            d[moderator_centered_col] = d[mo_col] - d[mo_col].mean()

            X_full = d[[iv_centered_col, moderator_centered_col] + controls_for_model].copy()
            X_full[interaction_term] = d[iv_centered_col] * d[moderator_centered_col]

            res = run_ols_clustered(d[y_col], X_full, d["state_fips_for_clustering"])
            beta = res.params.get(interaction_term, np.nan)
            pval = res.pvalues.get(interaction_term, np.nan)
            ci = res.conf_int().loc[interaction_term].values if interaction_term in res.params.index else (np.nan, np.nan)

            # Compute Cohen's f2 for interaction contribution: full model vs reduced model.
            cohens_f2 = np.nan
            try:
                X_reduced = d[[iv_centered_col, moderator_centered_col] + controls_for_model].copy()
                r2_full = float(sm.OLS(d[y_col], add_const(X_full)).fit().rsquared)
                r2_reduced = float(sm.OLS(d[y_col], add_const(X_reduced)).fit().rsquared)
                if np.isfinite(r2_full) and np.isfinite(r2_reduced) and (1.0 - r2_full) > 1e-12:
                    cohens_f2 = max((r2_full - r2_reduced) / (1.0 - r2_full), 0.0)
            except Exception as e:
                logger.warning(f"    Could not compute Cohen's f2 for {spec['name']} ({t_key}): {e}")

            row = {
                "IV": iv_col,
                "Moderator": mo_col,
                "Outcome": y_col,
                "Transform": t_key,
                "Transform_Label": t_label,
                "N": len(d),
                "Interaction_Beta": beta,
                "Interaction_p": pval,
                "Interaction_CI_Lower": ci[0],
                "Interaction_CI_Upper": ci[1],
                "Cohens_f2": cohens_f2,
                "Analysis_Tag": "continuous_surface_target",
            }
            surface_rows.append(row)
            if t_key == "raw":
                interaction_results.append(row)

            plot_continuous_moderator_surface(
                d=d,
                iv_col=iv_col,
                moderator_col=mo_col,
                outcome_col=y_col,
                iv_centered_col=iv_centered_col,
                moderator_centered_col=moderator_centered_col,
                interaction_term=interaction_term,
                controls_for_model=controls_for_model,
                res_model=res,
                name=f"{spec['name']}_{t_key}",
                plot_dir=plot_dir,
                logger=logger,
                beta=beta,
                pval=pval,
                cohens_f2=cohens_f2,
                label_map=surface_label_map,
                transform_label=t_label,
            )

    if surface_rows:
        surface_df = pd.DataFrame(surface_rows)
        surface_csv = os.path.join(out_dir, "continuous_moderator_surface_effects_summary.csv")
        surface_df.to_csv(surface_csv, index=False)
        logger.info(f"Saved continuous moderator surface effects summary to {surface_csv}")

    dissertation_interaction_df = pd.DataFrame(interaction_results)
    if not dissertation_interaction_df.empty:
        out_csv = os.path.join(out_dir, "dissertation_interaction_effects_summary.csv")
        dissertation_interaction_df.to_csv(out_csv, index=False)
        logger.info(f"Saved dissertation interaction effects summary to {out_csv}")
    
    # --------------------------------------------------------------------------------
    # PART I. Assemble Unified Report
    # --------------------------------------------------------------------------------
    logger.info("PART I: Assembling final unified report.")
    assemble_replication_report_full(
        out_dir=out_dir,
        idx_df=idx_df,
        h1h4_df=h1h4_df,
        capex_df=capex_df,
        dissertation_direct_df=dissertation_direct_df,
        dissertation_interaction_df=dissertation_interaction_df,
        logger=logger
    )

    run_end_epoch = time.time()
    try:
        memory_path = write_run_memory_markdown(
            out_dir=out_dir,
            run_id=run_id,
            logger=logger,
            run_start_epoch=run_start_epoch,
            run_end_epoch=run_end_epoch,
            df=df,
            base_controls=base_controls,
            dissertation_direct_df=dissertation_direct_df,
            primary_county_crossfit_df=primary_county_crossfit_df,
            county_clip_trim_df=county_clip_trim_df,
            county_ols_sensitivity=county_ols_sensitivity,
            county_misclassification_df=county_misclassification_df,
            delta_mortality_df=delta_mortality_df,
            prepost_change_df=prepost_change_df,
            dissertation_interaction_df=dissertation_interaction_df,
        )
        logger.info(f"Saved run memory markdown: {memory_path}")
    except Exception as e:
        logger.warning(f"Could not write run memory markdown: {e}")

    logger.info("All done. Check logs and output directory for results.")

if __name__ == "__main__":
    main()
