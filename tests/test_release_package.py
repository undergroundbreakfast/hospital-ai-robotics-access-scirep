from __future__ import annotations

import hashlib
from pathlib import Path

from conftest import REPO_ROOT


REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "FILE_MANIFEST_SHA256.txt",
    "code/replicate_scirep_outcomes.py",
    "code/geospatial_access_lorenz.py",
    "code/support/build_pre_exposure_balance.py",
    "code/support/build_system_membership_sensitivity.py",
    "code/support/build_organizational_capacity_sensitivity.py",
    "sql/views/public.vw_hospital_aipw_with_placebo.sql",
    "sql/views/public.vw_county_tech_summary_adjpd.sql",
    "results/tables/primary_county_crossfit_summary.csv",
    "results/tables/revision_diagnostics/table_s28_organizational_capacity_aipw_sensitivity.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_required_release_files_are_present() -> None:
    missing = [rel for rel in REQUIRED_FILES if not (REPO_ROOT / rel).exists()]
    assert not missing, f"Missing required release files: {missing}"


def test_manifest_hashes_match_current_files() -> None:
    manifest = REPO_ROOT / "FILE_MANIFEST_SHA256.txt"
    assert manifest.exists()
    checked = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected_hash, rel = line.split(maxsplit=1)
        rel = rel.removeprefix("./")
        path = REPO_ROOT / rel
        assert path.exists(), f"Manifest entry points to missing file: {rel}"
        assert sha256(path) == expected_hash, f"Manifest hash mismatch for {rel}"
        checked += 1
    assert checked >= 70


def test_public_package_does_not_include_restricted_raw_data() -> None:
    forbidden_patterns = [
        "data/raw",
        "data/public/cms_2022_hospital_quality/raw",
        "data/public/cms_2022_hospital_quality/archives",
        "data/public/cms_2022_hospital_quality/filtered",
    ]
    existing_forbidden = [
        rel
        for rel in forbidden_patterns
        if (REPO_ROOT / rel).exists() and any((REPO_ROOT / rel).rglob("*"))
    ]
    assert not existing_forbidden, f"Restricted/raw data should not be committed: {existing_forbidden}"

