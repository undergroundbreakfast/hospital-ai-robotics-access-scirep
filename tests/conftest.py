from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def table_path(relative_path: str) -> Path:
    return REPO_ROOT / "results" / "tables" / relative_path


def load_table(relative_path: str) -> pd.DataFrame:
    path = table_path(relative_path)
    assert path.exists(), f"Expected result table is missing: {path}"
    return pd.read_csv(path)


def one_row(df: pd.DataFrame, **filters) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, expected in filters.items():
        assert col in df.columns, f"Missing column {col!r}; columns={list(df.columns)}"
        mask &= df[col].eq(expected)
    matched = df.loc[mask]
    assert len(matched) == 1, f"Expected one row for {filters}, found {len(matched)}"
    return matched.iloc[0]


def assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
    assert math.isfinite(float(actual)), f"Non-finite value: {actual!r}"
    assert float(actual) == pytest.approx(expected, abs=tol)

