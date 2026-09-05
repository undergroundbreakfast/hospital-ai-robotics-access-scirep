#!/usr/bin/env python3
"""Build load-ready 2022 CMS hospital quality extracts for SEP-1 and pneumonia.

Inputs are official CMS Provider Data Catalog hospital archive zip files stored
under ./archives. Outputs are normalized CSVs under ./filtered plus raw copies of
the source CSVs needed for auditability.
"""

from __future__ import annotations

import csv
import argparse
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "data" / "public" / "cms_2022_hospital_quality"
ARCHIVE_DIR = ROOT / "archives"
RAW_DIR = ROOT / "raw"
FILTERED_DIR = ROOT / "filtered"

SNAPSHOTS = {
    "2022_01": "hospitals_01_2022.zip",
    "2022_04": "hospitals_04_2022.zip",
    "2022_07": "hospitals_07_2022.zip",
    "2022_10": "hospitals_10_2022.zip",
}

TARGET_FILES = {
    "timely": "Timely_and_Effective_Care-Hospital.csv",
    "complications": "Complications_and_Deaths-Hospital.csv",
    "general_info": "Hospital_General_Information.csv",
    "measure_dates": "Measure_Dates.csv",
}

TARGET_MEASURES = {
    "timely": {"SEP_1"},
    "complications": {"MORT_30_PN"},
}

OUTPUT_COLUMNS = [
    "snapshot",
    "source_archive",
    "source_file",
    "facility_id",
    "facility_name",
    "address",
    "city",
    "state",
    "zip_code",
    "county_name",
    "phone_number",
    "hospital_type",
    "hospital_ownership",
    "emergency_services",
    "condition",
    "measure_id",
    "measure_name",
    "score_raw",
    "score_numeric",
    "sample",
    "denominator",
    "compared_to_national",
    "lower_estimate",
    "higher_estimate",
    "footnote",
    "row_start_date",
    "row_end_date",
    "measure_start_quarter",
    "measure_start_date",
    "measure_end_quarter",
    "measure_end_date",
]


def find_member(zf: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"Could not find {suffix} in {zf.filename}")
    if len(matches) > 1:
        non_revised = [name for name in matches if "Revised" not in name]
        matches = non_revised or matches
    return matches[0]


def read_csv_from_zip(zf: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with zf.open(member) as fh:
        text = (line.decode("utf-8-sig", errors="replace") for line in fh)
        return list(csv.DictReader(text))


def copy_raw_member(zf: zipfile.ZipFile, member: str, snapshot: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{snapshot}_{Path(member).name}"
    with zf.open(member) as src, out_path.open("wb") as dst:
        dst.write(src.read())


def numeric_or_blank(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.lower() in {"not available", "not applicable"}:
        return ""
    try:
        return str(float(cleaned))
    except ValueError:
        return ""


def build_general_info(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        fid = row.get("Facility ID", "").strip()
        if not fid:
            continue
        out[fid] = {
            "hospital_type": row.get("Hospital Type", ""),
            "hospital_ownership": row.get("Hospital Ownership", ""),
            "emergency_services": row.get("Emergency Services", ""),
        }
    return out


def build_measure_dates(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        measure_id = row.get("Measure ID", "").strip()
        if measure_id in {"SEP_1", "MORT_30_PN"}:
            out[measure_id] = {
                "measure_start_quarter": row.get("Measure Start Quarter", ""),
                "measure_start_date": row.get("Start Date", ""),
                "measure_end_quarter": row.get("Measure End Quarter", ""),
                "measure_end_date": row.get("End Date", ""),
            }
    return out


def normalize_row(
    *,
    snapshot: str,
    source_archive: str,
    source_file: str,
    row: dict[str, str],
    general_info: dict[str, dict[str, str]],
    measure_dates: dict[str, dict[str, str]],
) -> dict[str, str]:
    facility_id = row.get("Facility ID", "").strip()
    measure_id = row.get("Measure ID", "").strip()
    score_raw = row.get("Score", "")

    out = {
        "snapshot": snapshot,
        "source_archive": source_archive,
        "source_file": source_file,
        "facility_id": facility_id,
        "facility_name": row.get("Facility Name", ""),
        "address": row.get("Address", ""),
        "city": row.get("City", ""),
        "state": row.get("State", ""),
        "zip_code": row.get("ZIP Code", ""),
        "county_name": row.get("County Name", ""),
        "phone_number": row.get("Phone Number", ""),
        "hospital_type": "",
        "hospital_ownership": "",
        "emergency_services": "",
        "condition": row.get("Condition", ""),
        "measure_id": measure_id,
        "measure_name": row.get("Measure Name", ""),
        "score_raw": score_raw,
        "score_numeric": numeric_or_blank(score_raw),
        "sample": row.get("Sample", ""),
        "denominator": row.get("Denominator", ""),
        "compared_to_national": row.get("Compared to National", ""),
        "lower_estimate": row.get("Lower Estimate", ""),
        "higher_estimate": row.get("Higher Estimate", ""),
        "footnote": row.get("Footnote", ""),
        "row_start_date": row.get("Start Date", ""),
        "row_end_date": row.get("End Date", ""),
        "measure_start_quarter": "",
        "measure_start_date": "",
        "measure_end_quarter": "",
        "measure_end_date": "",
    }
    out.update(general_info.get(facility_id, {}))
    out.update(measure_dates.get(measure_id, {}))
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def download_archives() -> None:
    """Download the four fixed CMS snapshots, keeping incomplete files separate."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in SNAPSHOTS.values():
        destination = ARCHIVE_DIR / filename
        if destination.exists():
            continue
        url = "https://data.cms.gov/provider-data/sites/default/files/archive/Hospitals/2022/" + filename
        temporary = destination.with_suffix(".zip.part")
        try:
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
            if not zipfile.is_zipfile(temporary):
                raise ValueError(f"CMS did not return a ZIP archive for {filename}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    all_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    for snapshot, archive_name in SNAPSHOTS.items():
        archive_path = ARCHIVE_DIR / archive_name
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing expected archive: {archive_path}")

        with zipfile.ZipFile(archive_path) as zf:
            members = {key: find_member(zf, suffix) for key, suffix in TARGET_FILES.items()}
            for member in members.values():
                copy_raw_member(zf, member, snapshot)

            general_info = build_general_info(read_csv_from_zip(zf, members["general_info"]))
            measure_dates = build_measure_dates(read_csv_from_zip(zf, members["measure_dates"]))

            for source_key in ("timely", "complications"):
                rows = read_csv_from_zip(zf, members[source_key])
                target_ids = TARGET_MEASURES[source_key]
                filtered = [
                    normalize_row(
                        snapshot=snapshot,
                        source_archive=archive_name,
                        source_file=members[source_key],
                        row=row,
                        general_info=general_info,
                        measure_dates=measure_dates,
                    )
                    for row in rows
                    if row.get("Measure ID", "").strip() in target_ids
                ]
                all_rows.extend(filtered)

                for measure_id in sorted(target_ids):
                    measure_rows = [row for row in filtered if row["measure_id"] == measure_id]
                    out_name = f"cms_{snapshot}_{measure_id.lower()}_hospital.csv"
                    write_csv(FILTERED_DIR / out_name, measure_rows)
                    manifest_rows.append(
                        {
                            "snapshot": snapshot,
                            "source_archive": archive_name,
                            "source_file": members[source_key],
                            "measure_id": measure_id,
                            "row_count": str(len(measure_rows)),
                            "nonmissing_score_count": str(
                                sum(1 for row in measure_rows if row["score_numeric"])
                            ),
                        }
                    )

    write_csv(FILTERED_DIR / "cms_2022_hospital_quality_long.csv", all_rows)
    write_csv(
        FILTERED_DIR / "cms_2022_sep1_hospital_long.csv",
        [row for row in all_rows if row["measure_id"] == "SEP_1"],
    )
    write_csv(
        FILTERED_DIR / "cms_2022_mort30pn_hospital_long.csv",
        [row for row in all_rows if row["measure_id"] == "MORT_30_PN"],
    )

    manifest_path = FILTERED_DIR / "cms_2022_extract_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "snapshot",
                "source_archive",
                "source_file",
                "measure_id",
                "row_count",
                "nonmissing_score_count",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(all_rows)} combined rows to {FILTERED_DIR / 'cms_2022_hospital_quality_long.csv'}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing fixed CMS 2022 archives before extraction")
    args = parser.parse_args()
    if args.download:
        download_archives()
    main()
