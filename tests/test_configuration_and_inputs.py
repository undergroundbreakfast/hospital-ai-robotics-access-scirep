from __future__ import annotations
import importlib.util
import io
from pathlib import Path
import sys
import zipfile
import pytest
from conftest import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "code"))
from scirep_config import database_settings, create_db_engine


def test_standard_database_settings_and_url_punctuation(monkeypatch):
    env = dict(PGHOST="database.example", PGPORT="5544", PGDATABASE="study", PGUSER="analyst", PGPASSWORD="dummy:@/#?pass")
    for name in ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD", "POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRESQL_KEY"]:
        monkeypatch.delenv(name, raising=False)
    for k,v in env.items(): monkeypatch.setenv(k,v)
    settings = database_settings()
    assert settings["host"] == "database.example" and settings["port"] == 5544
    # URL construction does not contact a database; stub the engine factory.
    import sqlalchemy
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda url, **kwargs: url)
    url = create_db_engine()
    assert url.password == env["PGPASSWORD"]
    assert url.database == "study" and url.username == "analyst" and url.port == 5544


def test_legacy_settings_and_standard_precedence():
    env = dict(POSTGRES_HOST="legacy", POSTGRES_PORT="5555", POSTGRES_DB="old", POSTGRES_USER="old_user", POSTGRESQL_KEY="dummy")
    assert database_settings(env)["host"] == "legacy"
    assert database_settings({**env, "PGHOST":"standard"})["host"] == "standard"
    with pytest.raises(ValueError, match="PGPASSWORD"): database_settings({})
    with pytest.raises(ValueError, match="PGPORT"): database_settings({**env,"PGPORT":"99999"})


def extractor():
    spec=importlib.util.spec_from_file_location("cms_extractor",REPO_ROOT/"code/support/build_cms_2022_quality_extract.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_cms_extractor_uses_documented_data_root():
    m=extractor()
    expected=REPO_ROOT/"data/public/cms_2022_hospital_quality"
    assert m.ARCHIVE_DIR == expected/"archives"
    assert m.FILTERED_DIR == expected/"filtered"


def test_cms_extraction_with_synthetic_archives(tmp_path,monkeypatch):
    m=extractor()
    for name,folder in [("ARCHIVE_DIR","archives"),("RAW_DIR","raw"),("FILTERED_DIR","filtered")]:
        monkeypatch.setattr(m,name,tmp_path/folder)
    m.ARCHIVE_DIR.mkdir()
    files={
        "Hospital_General_Information.csv":"Facility ID,Facility Name\n000001,Synthetic Hospital\n",
        "Measure_Dates.csv":"Measure ID,Measure Name\nSEP_1,Sepsis\nMORT_30_PN,Pneumonia\n",
        "Timely_and_Effective_Care-Hospital.csv":"Facility ID,Measure ID,Score\n000001,SEP_1,60\n",
        "Complications_and_Deaths-Hospital.csv":"Facility ID,Measure ID,Score\n000001,MORT_30_PN,16\n",
    }
    for filename in m.SNAPSHOTS.values():
        with zipfile.ZipFile(m.ARCHIVE_DIR/filename,"w") as z:
            for name,body in files.items():z.writestr(name,body)
    m.main()
    import csv
    rows=list(csv.DictReader((m.FILTERED_DIR/"cms_2022_hospital_quality_long.csv").open()))
    assert len(rows)==8
    assert {r["facility_id"] for r in rows}=={"000001"}
    assert {float(r["score_numeric"]) for r in rows}=={60.,16.}


def test_missing_schema_definitions_are_included():
    text=(REPO_ROOT/"sql/build_views.psql.sql").read_text()
    for name in ["vw_hospital_level_aipw","vw_county_file_export_wide","vw_conceptual_model_adjpd"]:
        assert "views/public."+name+".sql" in text
    assert "functions/public.safe_to_numeric.sql" in text
    assert text.index("BEGIN;") < text.index("COMMIT;")
