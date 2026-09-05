#!/usr/bin/env python3
"""Load CMS HCRIS 2023 Hospital Provider Cost Report financial fields.

The CMS Provider Cost Report PUF is a compact public-use projection of HCRIS.
This loader keeps the full 2023 PUF as text columns with normalized names and
creates a derived view for financial-capacity proxies used in reviewer-response
diagnostics:

* operating_margin = net income from service to patients / net patient revenue
* total_margin = net income / total income
* days_cash_on_hand = cash plus temporary investments divided by report-period
  daily cash operating expense

The view is keyed by CMS Certification Number (CCN), which joins cleanly to the
AHA Medicare provider number / CMS facility id used in the manuscript pipeline.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import re
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirep_config import connect_db, output_directory

ROOT = output_directory("hcris_2023")

import psycopg
from psycopg import sql

DATASET_UUID = "44060663-47d8-4ced-a115-b53b4c270acb"
DATA_API_URL = (
    f"https://data.cms.gov/data-api/v1/dataset/{DATASET_UUID}/data-viewer?size=1"
)
FALLBACK_CSV_URL = (
    "https://data.cms.gov/sites/default/files/2026-01/"
    "3c39f483-c7e0-4025-8396-4df76942e10f/CostReport_2023_Final.csv"
)
TABLE_NAME = "cms_hcris_2023_cost_report_puf"
VIEW_NAME = "vw_cms_hcris_2023_financial_capacity"
DERIVED_TABLE_NAME = "cms_hcris_2023_financial_capacity"


def connect():
    return connect_db()


def snake_case(name: str) -> str:
    out = name.strip().lower()
    out = out.replace("&", " and ")
    out = re.sub(r"[^a-z0-9]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        out = "unnamed"
    if out[0].isdigit():
        out = f"col_{out}"
    return out


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = snake_case(name)
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count + 1}")
    return out


def resolve_csv_url() -> str:
    try:
        with urllib.request.urlopen(DATA_API_URL, timeout=30) as response:
            import json

            payload = json.load(response)
        path = payload.get("meta", {}).get("data_file_url")
        if path:
            return f"https://data.cms.gov{path}" if path.startswith("/") else path
    except Exception as exc:
        print(f"Warning: could not resolve CMS data_file_url from API ({exc}); using fallback.")
    return FALLBACK_CSV_URL


def download_csv(url: str, raw_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, raw_path.open("wb") as out:
        out.write(response.read())


def normalize_csv(raw_path: Path, normalized_path: Path) -> tuple[list[str], int]:
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.reader(src)
        original_header = next(reader)
        header = unique_names(original_header)
        with normalized_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(header)
            row_count = 0
            for row in reader:
                writer.writerow(row)
                row_count += 1
    return header, row_count


def create_table(conn: psycopg.Connection, columns: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS public.{} ({})").format(
                sql.Identifier(TABLE_NAME),
                sql.SQL(", ").join(
                    sql.SQL("{} text").format(sql.Identifier(col)) for col in columns
                ),
            )
        )
        cur.execute(sql.SQL("TRUNCATE TABLE public.{}").format(sql.Identifier(TABLE_NAME)))
        cur.execute(
            sql.SQL("COMMENT ON TABLE public.{} IS {}").format(
                sql.Identifier(TABLE_NAME),
                sql.Literal(
                    "CMS Hospital Provider Cost Report 2023 PUF loaded from data.cms.gov; "
                    "columns normalized to snake_case and stored as text."
                ),
            )
        )


def copy_csv(conn: psycopg.Connection, columns: list[str], normalized_path: Path) -> None:
    with conn.cursor() as cur, normalized_path.open("r", encoding="utf-8") as src:
        copy_sql = sql.SQL("COPY public.{} ({}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')").format(
            sql.Identifier(TABLE_NAME),
            sql.SQL(", ").join(sql.Identifier(col) for col in columns),
        )
        with cur.copy(copy_sql) as copy:
            while chunk := src.read(1024 * 1024):
                copy.write(chunk)


def create_view(conn: psycopg.Connection) -> None:
    number_expr = (
        "nullif(regexp_replace({col}::text, '[^0-9.\\-]', '', 'g'), '')::numeric"
    )

    def n(col: str) -> str:
        return number_expr.format(col=col)

    view_sql = f"""
    create or replace view public.{VIEW_NAME} as
    with raw_typed as (
        select
            rpt_rec_num,
            lpad(provider_ccn::text, 6, '0') as provider_ccn,
            hospital_name,
            state_code,
            county,
            medicare_cbsa_number,
            rural_versus_urban,
            ccn_facility_type,
            provider_type,
            type_of_control,
            fiscal_year_begin_date::date as fiscal_year_begin_date,
            fiscal_year_end_date::date as fiscal_year_end_date,
            (fiscal_year_end_date::date - fiscal_year_begin_date::date + 1) as report_days,
            {n("cash_on_hand_and_in_banks")} as cash_on_hand_and_in_banks,
            {n("temporary_investments")} as temporary_investments,
            {n("total_current_assets")} as total_current_assets,
            {n("total_assets")} as total_assets,
            {n("total_current_liabilities")} as total_current_liabilities,
            {n("total_liabilities")} as total_liabilities,
            {n("total_patient_revenue")} as total_patient_revenue,
            {n("net_patient_revenue")} as net_patient_revenue,
            {n("less_total_operating_expense")} as total_operating_expense,
            {n("depreciation_cost")} as depreciation_cost,
            {n("net_income_from_service_to_patients")} as net_income_from_service_to_patients,
            {n("total_other_income")} as total_other_income,
            {n("total_income")} as total_income,
            {n("total_other_expenses")} as total_other_expenses,
            {n("net_income")} as net_income,
            {n("cost_to_charge_ratio")} as cost_to_charge_ratio,
            row_number() over (
                partition by lpad(provider_ccn::text, 6, '0')
                order by fiscal_year_end_date::date desc nulls last,
                         (fiscal_year_end_date::date - fiscal_year_begin_date::date + 1) desc nulls last,
                         rpt_rec_num desc
            ) as ccn_report_rank,
            count(*) over (partition by lpad(provider_ccn::text, 6, '0')) as ccn_report_count
        from public.{TABLE_NAME}
    ),
    typed as (
        select *
        from raw_typed
        where ccn_report_rank = 1
    ),
    derived as (
        select
            *,
            coalesce(cash_on_hand_and_in_banks, 0) + coalesce(temporary_investments, 0)
                as cash_and_temporary_investments,
            total_operating_expense - coalesce(depreciation_cost, 0) as cash_operating_expense,
            net_patient_revenue + coalesce(total_other_income, 0) as total_revenue_for_margin
        from typed
    )
    select
        *,
        case
            when net_patient_revenue > 0
            then net_income_from_service_to_patients / net_patient_revenue
        end as operating_margin,
        case
            when total_revenue_for_margin > 0
            then net_income / total_revenue_for_margin
        end as total_margin,
        case
            when report_days > 0 and cash_operating_expense > 0
             and cash_and_temporary_investments >= 0
            then cash_and_temporary_investments / (cash_operating_expense / report_days)
        end as days_cash_on_hand,
        case
            when total_current_liabilities > 0
            then total_current_assets / total_current_liabilities
        end as current_ratio,
        case
            when net_patient_revenue > 0
            then cash_and_temporary_investments / net_patient_revenue
        end as cash_to_net_patient_revenue
    from derived;

    create index if not exists idx_{TABLE_NAME}_provider_ccn
        on public.{TABLE_NAME} (provider_ccn);

    comment on view public.{VIEW_NAME} is
        'Derived CMS HCRIS 2023 financial-capacity proxies by CCN: operating margin, total margin, days cash on hand, current ratio, and cash-to-net-patient-revenue.';
    """
    with conn.cursor() as cur:
        cur.execute(f"drop view if exists public.{VIEW_NAME}")
        cur.execute(view_sql)
        cur.execute(f"drop table if exists public.{DERIVED_TABLE_NAME}")
        cur.execute(
            f"create table public.{DERIVED_TABLE_NAME} as "
            f"select * from public.{VIEW_NAME}"
        )
        cur.execute(
            f"create unique index idx_{DERIVED_TABLE_NAME}_provider_ccn "
            f"on public.{DERIVED_TABLE_NAME} (provider_ccn)"
        )
        cur.execute(
            f"comment on table public.{DERIVED_TABLE_NAME} is "
            "'Materialized CMS HCRIS 2023 financial-capacity proxies by CCN, "
            "derived from public.cms_hcris_2023_cost_report_puf.'"
        )


def main() -> None:
    raw_path = ROOT / "raw" / "CostReport_2023_Final.csv"
    normalized_path = ROOT / "processed" / "cms_hcris_2023_cost_report_puf_normalized.csv"
    url_path = ROOT / "processed" / "source_url.txt"
    csv_url = resolve_csv_url()
    print(f"Downloading CMS HCRIS 2023 PUF from {csv_url}")
    download_csv(csv_url, raw_path)
    columns, row_count = normalize_csv(raw_path, normalized_path)
    url_path.parent.mkdir(parents=True, exist_ok=True)
    url_path.write_text(
        f"{csv_url}\nDownloaded: {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
        f"Rows: {row_count}\nColumns: {len(columns)}\n",
        encoding="utf-8",
    )

    with connect() as conn:
        create_table(conn, columns)
        copy_csv(conn, columns, normalized_path)
        create_view(conn)
        conn.commit()

    print(f"Loaded {row_count:,} rows into public.{TABLE_NAME}.")
    print(f"Created/updated public.{VIEW_NAME}.")
    print(f"Created/updated public.{DERIVED_TABLE_NAME}.")
    print(f"Raw CSV: {raw_path}")
    print(f"Normalized CSV: {normalized_path}")


if __name__ == "__main__":
    main()
