#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2026 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
# -*- coding: utf-8 -*-
"""
Focused Moderation Visualization Script (v1)

Purpose:
- Preserve v55 operational patterns (database connection, logging, run folders, memory file)
- Generate moderation plots for:
    IV3 (Community Health Behaviors) -> DV21 (Premature Mortality YPLL)
    grouped by MO14 (AI routine-task automation access)
- Produce four visualization variants to assess skew/zero inflation sensitivity:
    raw, winsorized, log-transformed, z-scored
"""

import os
import sys
import uuid
import time
import warnings
import logging
import datetime
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:
    raise ImportError("matplotlib and seaborn are required for plotting.") from exc


def configure_warning_filters() -> None:
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in matmul")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")


configure_warning_filters()


def generate_run_id(run_start_dt: Optional[datetime.datetime] = None) -> str:
    ts = (run_start_dt or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def setup_logger(
    log_file_name_prefix: str = "moderation_iv3_dv21_log",
    log_dir: str = "logs",
    run_id: Optional[str] = None,
):
    logger = logging.getLogger("moderation_iv3_dv21")
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

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logging to: {log_path}")
    return logger


def connect_to_database(logger) -> Engine:
    host = os.getenv("POSTGRES_HOST", "localhost")
    database = os.getenv("POSTGRES_DB", "Research_TEST")
    user = os.getenv("POSTGRES_USER", "postgres")
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
                text(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema IN ('public')
                      AND table_name = :tname
                    LIMIT 1
                    """
                ),
                {"tname": table_name},
            ).fetchone()
        return res is not None
    except Exception as e:
        logger.warning(f"Table existence check failed for {table_name}: {e}")
        return False


def list_columns(engine: Engine, table_name: str, logger) -> List[str]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :tname
                    """
                ),
                {"tname": table_name},
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


def resolve_first_existing_column(available_cols: List[str], alias_list: List[str]) -> Optional[str]:
    for col in alias_list:
        if col in available_cols:
            return col
    return None


def fetch_focus_data(engine: Engine, logger) -> pd.DataFrame:
    vcm_table = choose_first_existing_table(
        engine,
        ["vw_conceptual_model_adjpd", "vw_conceptual_model"],
        logger,
    )
    vcv_table = choose_first_existing_table(
        engine,
        ["vw_conceptual_model_variables_adjpd", "vw_conceptual_model_variables"],
        logger,
    )
    vcts_table = choose_first_existing_table(
        engine,
        ["vw_adjpd_weighted_tech_summary", "vw_county_tech_summary_adjpd", "vw_county_tech_summary"],
        logger,
    )

    if vcm_table is None or vcv_table is None or vcts_table is None:
        sys.exit("Required source views are missing. Aborting.")

    mo14_aliases = ["pct_wfaiart_enabled_adjpd", "pct_wfaiart_enabled"]
    tech_cols_available = list_columns(engine, vcts_table, logger)
    mo14_col = resolve_first_existing_column(tech_cols_available, mo14_aliases)
    if mo14_col is None:
        sys.exit(f"Could not locate MO14 column in {vcts_table}. Tried: {mo14_aliases}")

    sql = f"""
        SELECT
            vcm.county_fips,
            vcm.health_behaviors_score AS iv3_health_behaviors_score,
            vcv.premature_death_raw_value AS dv21_premature_death_ypll_rate,
            vcts.{mo14_col} AS mo14_ai_automate_routine_tasks_pct
        FROM public.{vcm_table} AS vcm
        LEFT JOIN public.{vcv_table} AS vcv
          ON vcm.county_fips = vcv.county_fips
        LEFT JOIN public.{vcts_table} AS vcts
          ON vcm.county_fips = vcts.county_fips
        WHERE vcm.population IS NOT NULL
          AND CAST(vcm.population AS NUMERIC) > 0
    """

    try:
        df = pd.read_sql_query(text(sql), engine)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    for col in ["iv3_health_behaviors_score", "dv21_premature_death_ypll_rate", "mo14_ai_automate_routine_tasks_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    before = len(df)
    df = df[~df["county_fips"].str.endswith("000")].copy()
    dropped_non_county = before - len(df)
    if dropped_non_county > 0:
        logger.info(f"Dropped {dropped_non_county} non-county FIPS rows ending in '000'.")

    df = df.dropna(subset=["iv3_health_behaviors_score", "dv21_premature_death_ypll_rate", "mo14_ai_automate_routine_tasks_pct"]).copy()
    logger.info(f"Focused data prepared with N={len(df)} counties.")
    return df


def _winsorize_series(s: pd.Series, lo_q: float = 0.01, hi_q: float = 0.99) -> pd.Series:
    lo = s.quantile(lo_q)
    hi = s.quantile(hi_q)
    return s.clip(lower=lo, upper=hi)


def _zscore_series(s: pd.Series) -> pd.Series:
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def create_mo14_groups(df: pd.DataFrame, logger) -> Tuple[pd.Series, Dict[str, Any]]:
    mo14 = df["mo14_ai_automate_routine_tasks_pct"]
    zero_share = float((mo14 == 0).mean())

    if zero_share >= 0.15:
        group = np.where(mo14 > 0, "High AI Access (Any)", "Low AI Access (None)")
        rule = "any_vs_none"
        threshold = 0.0
    else:
        threshold = float(mo14.median())
        group = np.where(mo14 >= threshold, "High AI Access (>= Median)", "Low AI Access (< Median)")
        rule = "median_split"

    group_s = pd.Series(group, index=df.index, name="mo14_group")
    counts = group_s.value_counts(dropna=False).to_dict()

    logger.info(f"MO14 grouping rule: {rule}; zero_share={zero_share:.3f}; threshold={threshold:.4f}")
    logger.info(f"Group counts: {counts}")

    return group_s, {
        "rule": rule,
        "zero_share": zero_share,
        "threshold": threshold,
        "counts": counts,
    }


def make_variant_dataframe(df: pd.DataFrame, variant: str, logger) -> Tuple[pd.DataFrame, str, str]:
    out = df.copy()
    x_col = "iv3_health_behaviors_score"
    y_col = "dv21_premature_death_ypll_rate"

    if variant == "raw":
        out["x_plot"] = out[x_col]
        out["y_plot"] = out[y_col]
        x_label = "Community Health Behaviors Index\n(Higher = More Smoking, Obesity, Inactivity)"
        y_label = "Premature Mortality (YPLL per 100k)"
    elif variant == "winsorized":
        out["x_plot"] = _winsorize_series(out[x_col])
        out["y_plot"] = _winsorize_series(out[y_col])
        x_label = "Community Health Behaviors Index (Winsorized 1%-99%)"
        y_label = "Premature Mortality (YPLL per 100k, Winsorized 1%-99%)"
    elif variant == "log1p":
        before = len(out)
        out = out[(out[x_col] >= 0) & (out[y_col] >= 0)].copy()
        logger.info(f"Log variant dropped {before - len(out)} rows with negative IV3 or DV21 values.")
        out["x_plot"] = np.log1p(out[x_col])
        out["y_plot"] = np.log1p(out[y_col])
        x_label = "log1p(Community Health Behaviors Index)"
        y_label = "log1p(Premature Mortality YPLL per 100k)"
    elif variant == "zscore":
        out["x_plot"] = _zscore_series(out[x_col])
        out["y_plot"] = _zscore_series(out[y_col])
        x_label = "Community Health Behaviors (Z-score)"
        y_label = "Premature Mortality YPLL (Z-score)"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_plot", "y_plot", "mo14_group"]).copy()
    return out, x_label, y_label


def _fit_line(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def plot_moderation_variant(
    d: pd.DataFrame,
    variant: str,
    out_path: str,
    x_label: str,
    y_label: str,
    logger,
) -> pd.DataFrame:
    color_map = {
        "Low AI Access (None)": "#d62728",
        "High AI Access (Any)": "#0b4f9c",
        "Low AI Access (< Median)": "#d62728",
        "High AI Access (>= Median)": "#0b4f9c",
    }

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_facecolor("#f2f2f2")
    plt.grid(True, linestyle="--", alpha=0.35)

    order = [g for g in ["Low AI Access (None)", "Low AI Access (< Median)", "High AI Access (Any)", "High AI Access (>= Median)"] if g in d["mo14_group"].unique()]

    line_rows: List[Dict[str, Any]] = []
    for group_name in order:
        sub = d[d["mo14_group"] == group_name]
        color = color_map[group_name]

        plt.scatter(
            sub["x_plot"],
            sub["y_plot"],
            s=12,
            alpha=0.18,
            color=color,
            edgecolors="none",
        )

        sns.regplot(
            data=sub,
            x="x_plot",
            y="y_plot",
            scatter=False,
            ci=95,
            color=color,
            line_kws={"linewidth": 2.8},
        )

        slope, intercept = _fit_line(sub["x_plot"], sub["y_plot"])
        line_rows.append(
            {
                "variant": variant,
                "group": group_name,
                "n": int(len(sub)),
                "slope": slope,
                "intercept": intercept,
            }
        )

    plt.title(
        "Risk Attenuation: AI Access Moderates the Link\n"
        "Between Unhealthy Behaviors and Premature Death",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    handles = [
        plt.Line2D([0], [0], color=color_map[g], lw=3, label=g)
        for g in order
    ]
    plt.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved {variant} moderation plot: {out_path}")
    return pd.DataFrame(line_rows)


def write_run_memory_markdown(
    out_dir: str,
    run_id: str,
    run_start_epoch: float,
    run_end_epoch: float,
    grouping_meta: Dict[str, Any],
    variant_results_df: pd.DataFrame,
    logger,
) -> str:
    memory_path = os.path.join(out_dir, f"run_memory_{run_id}.md")
    run_start_dt = datetime.datetime.fromtimestamp(run_start_epoch)
    run_end_dt = datetime.datetime.fromtimestamp(run_end_epoch)

    lines: List[str] = []
    lines.append(f"# Run Memory - {run_id}")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- Run_ID: `{run_id}`")
    lines.append(f"- Run_Directory: `{os.path.relpath(out_dir, os.getcwd())}`")
    lines.append(f"- Start_Time: `{run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- End_Time: `{run_end_dt.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Duration_Seconds: `{run_end_epoch - run_start_epoch:.2f}`")
    lines.append("")

    lines.append("## Moderation Setup")
    lines.append("- IV: `iv3_health_behaviors_score`")
    lines.append("- DV: `dv21_premature_death_ypll_rate`")
    lines.append("- Moderator (grouping source): `mo14_ai_automate_routine_tasks_pct`")
    lines.append(f"- Grouping_Rule: `{grouping_meta.get('rule')}`")
    lines.append(f"- MO14_Zero_Share: `{grouping_meta.get('zero_share', np.nan):.4f}`")
    lines.append(f"- MO14_Threshold: `{grouping_meta.get('threshold', np.nan):.4f}`")
    lines.append(f"- Group_Counts: `{grouping_meta.get('counts', {})}`")
    lines.append("")

    lines.append("## Slope Summary by Variant")
    if variant_results_df.empty:
        lines.append("_No model summary rows generated._")
    else:
        cols = ["variant", "group", "n", "slope", "intercept"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in variant_results_df.iterrows():
            lines.append(
                "| " + " | ".join(
                    [
                        str(row["variant"]),
                        str(row["group"]),
                        str(int(row["n"])),
                        f"{float(row['slope']):.6f}" if pd.notna(row["slope"]) else "NA",
                        f"{float(row['intercept']):.6f}" if pd.notna(row["intercept"]) else "NA",
                    ]
                ) + " |"
            )

    with open(memory_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Saved run memory markdown: {memory_path}")
    return memory_path


def main() -> None:
    run_start_dt = datetime.datetime.now()
    run_start_epoch = time.time()
    run_id = generate_run_id(run_start_dt)

    output_root_dir = "moderation_output_health_behaviors"
    out_dir = os.path.join(output_root_dir, "runs", run_id)
    plot_dir = os.path.join(out_dir, "plots")
    log_dir = os.path.join(out_dir, "logs")

    logger = setup_logger(log_dir=log_dir, run_id=run_id)
    logger.info("Starting focused moderation visualization script (v56).")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run output directory: {out_dir}")

    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    engine = connect_to_database(logger)
    df = fetch_focus_data(engine, logger)

    df["mo14_group"], grouping_meta = create_mo14_groups(df, logger)

    variants = ["raw", "winsorized", "log1p", "zscore"]
    summary_rows: List[pd.DataFrame] = []

    for variant in variants:
        d_variant, x_label, y_label = make_variant_dataframe(df, variant, logger)
        plot_path = os.path.join(plot_dir, f"moderation_iv3_dv21_mo14_{variant}.png")
        line_df = plot_moderation_variant(
            d=d_variant,
            variant=variant,
            out_path=plot_path,
            x_label=x_label,
            y_label=y_label,
            logger=logger,
        )
        summary_rows.append(line_df)

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    summary_csv = os.path.join(out_dir, "moderation_variant_line_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved slope summary: {summary_csv}")

    preview_cols = [
        "county_fips",
        "iv3_health_behaviors_score",
        "dv21_premature_death_ypll_rate",
        "mo14_ai_automate_routine_tasks_pct",
        "mo14_group",
    ]
    preview_csv = os.path.join(out_dir, "moderation_focus_dataset_preview.csv")
    df[preview_cols].to_csv(preview_csv, index=False)
    logger.info(f"Saved focused dataset preview: {preview_csv}")

    run_end_epoch = time.time()
    write_run_memory_markdown(
        out_dir=out_dir,
        run_id=run_id,
        run_start_epoch=run_start_epoch,
        run_end_epoch=run_end_epoch,
        grouping_meta=grouping_meta,
        variant_results_df=summary_df,
        logger=logger,
    )

    logger.info("All done. Check run folder for plots, summaries, and memory file.")


if __name__ == "__main__":
    main()
