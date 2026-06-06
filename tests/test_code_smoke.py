from __future__ import annotations

import py_compile
from pathlib import Path

from conftest import REPO_ROOT


def test_python_scripts_compile() -> None:
    scripts = sorted((REPO_ROOT / "code").glob("*.py")) + sorted(
        (REPO_ROOT / "code" / "support").glob("*.py")
    )
    assert scripts, "No Python scripts found to validate"
    for script in scripts:
        py_compile.compile(str(script), doraise=True)


def test_sql_views_are_read_only_definitions() -> None:
    sql_files = sorted((REPO_ROOT / "sql" / "views").glob("*.sql"))
    assert len(sql_files) >= 10
    forbidden = {"insert ", "update ", "delete ", "drop table", "truncate "}
    for path in sql_files:
        text = path.read_text().lower()
        assert "create" in text and "view" in text, f"Expected view definition in {path}"
        hits = [token for token in forbidden if token in text]
        assert not hits, f"Unexpected mutating SQL in {path}: {hits}"

