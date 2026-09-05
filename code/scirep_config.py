"""Shared connection settings and local output paths for the research workflows."""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = REPO_ROOT / "outputs"
PRIVATE_DIR = REPO_ROOT / "data" / "restricted"


def database_settings(environ=None):
    env = os.environ if environ is None else environ
    def value(primary, legacy, default=None):
        return env.get(primary) or env.get(legacy) or default
    password = value("PGPASSWORD", "POSTGRESQL_KEY")
    if not password:
        raise ValueError("Set PGPASSWORD (or POSTGRESQL_KEY) before running database workflows.")
    port = int(value("PGPORT", "POSTGRES_PORT", "5432"))
    if not 1 <= port <= 65535:
        raise ValueError("PGPORT must be between 1 and 65535.")
    return dict(host=value("PGHOST", "POSTGRES_HOST", "localhost"),
                port=port, dbname=value("PGDATABASE", "POSTGRES_DB", "Research_TEST"),
                user=value("PGUSER", "POSTGRES_USER", "postgres"), password=password)


def create_db_engine():
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL
    settings = database_settings()
    settings["database"] = settings.pop("dbname")
    settings["username"] = settings.pop("user")
    url = URL.create("postgresql+psycopg2", **settings)
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})


def connect_db():
    import psycopg
    return psycopg.connect(**database_settings(), connect_timeout=10)


def output_directory(name):
    path = GENERATED_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def restricted_directory(name):
    path = PRIVATE_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
