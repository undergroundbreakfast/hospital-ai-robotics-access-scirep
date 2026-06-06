from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.db


REQUIRED_OBJECTS = {
    "aha_fy2022",
    "aha_fy2023",
    "aha_fy2024",
    "cms_2022_hospital_quality_long",
    "cms_hcris_2023_financial_capacity",
    "vw_hospital_level_aipw",
    "vw_county_file_export_wide",
}

REQUIRED_COLUMNS = {
    "aha_fy2024": {
        "id",
        "mcrnum",
        "fcounty",
        "robohos",
        "mhsmemb",
        "sysid",
        "adjpd",
        "bdtot",
        "ften",
        "vrn",
        "cbsacode",
    },
    "cms_2022_hospital_quality_long": {"facility_id", "snapshot", "measure_id", "score_numeric"},
    "cms_hcris_2023_financial_capacity": {"provider_ccn"},
    "vw_hospital_level_aipw": {"aha_id", "cms_facility_id", "wfaiss", "wfaiart", "robohos"},
    "vw_county_file_export_wide": {
        "county_fips",
        "dv21_premature_death_ypll_rate",
        "ct6_hospital_deaths_age_adj_2023",
        "mo14_ai_automate_routine_tasks_pct",
        "mo21_robotics_in_hospital_pct",
    },
}

OPTIONAL_FULL_REPRODUCTION_COLUMNS = {
    "aha_fy2024": {"ftern", "crnfte", "ceamt", "gfeet"},
}


def _db_enabled() -> bool:
    return os.getenv("RUN_DB_TESTS") == "1"


def _connect():
    if not _db_enabled():
        pytest.skip("Set RUN_DB_TESTS=1 and Postgres credentials to run database contract tests.")
    try:
        import psycopg
    except ImportError:
        try:
            import psycopg2 as psycopg
        except ImportError as exc:
            pytest.skip(f"Neither psycopg nor psycopg2 is installed: {exc}")
    password = os.getenv("POSTGRESQL_KEY") or os.getenv("PGPASSWORD")
    if not password:
        pytest.skip("Set POSTGRESQL_KEY or PGPASSWORD to run database contract tests.")
    conn = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "localhost")),
        port=int(os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432"))),
        dbname=os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "Research_TEST")),
        user=os.getenv("POSTGRES_USER", os.getenv("PGUSER", "postgres")),
        password=password,
        connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute("set statement_timeout = '15s'")
    return conn


def test_required_database_objects_exist() -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select table_name
            from information_schema.tables
            where table_schema = 'public'
              and table_name = any(%s)
            """,
            (sorted(REQUIRED_OBJECTS),),
        )
        found = {row[0] for row in cur.fetchall()}
    assert REQUIRED_OBJECTS <= found


@pytest.mark.parametrize("object_name, required_columns", sorted(REQUIRED_COLUMNS.items()))
def test_required_database_columns_exist(object_name: str, required_columns: set[str]) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'public'
              and table_name = %s
            """,
            (object_name,),
        )
        found = {row[0] for row in cur.fetchall()}
    assert required_columns <= found


def test_public_quality_and_hcris_inputs_have_rows() -> None:
    checks = [
        "select 1 from public.cms_2022_hospital_quality_long where measure_id in ('SEP_1', 'MORT_30_PN') limit 1",
        "select 1 from public.cms_hcris_2023_financial_capacity limit 1",
        "select 1 from public.aha_fy2024 where robohos is not null or mhsmemb is not null or adjpd is not null limit 1",
    ]
    with _connect() as conn, conn.cursor() as cur:
        for sql in checks:
            cur.execute(sql)
            assert cur.fetchone() is not None, f"No rows returned for: {sql}"


@pytest.mark.parametrize("object_name, optional_columns", sorted(OPTIONAL_FULL_REPRODUCTION_COLUMNS.items()))
def test_optional_full_reproduction_raw_columns_are_available(
    object_name: str, optional_columns: set[str]
) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select column_name
            from information_schema.columns
            where table_schema = 'public'
              and table_name = %s
            """,
            (object_name,),
        )
        found = {row[0] for row in cur.fetchall()}
    missing = optional_columns - found
    if missing:
        pytest.xfail(
            "This database has the released result artifacts and analysis views, "
            f"but not every raw AHA field needed for full organizational-capacity reruns: {sorted(missing)}"
        )
