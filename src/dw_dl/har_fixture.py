"""Bounded raw-data fixture extraction for real-data smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


GROUP_COLUMNS = ("User", "Model", "Device")
CSV_COLUMNS = (
    "Index",
    "Arrival_Time",
    "Creation_Time",
    "x",
    "y",
    "z",
    "User",
    "Model",
    "Device",
    "gt",
)
SENSOR_FILES = {
    "acc": ("Phones_accelerometer.csv", "Watch_accelerometer.csv"),
    "gyro": ("Phones_gyroscope.csv", "Watch_gyroscope.csv"),
}


@dataclass(frozen=True)
class FixtureLimits:
    chunk_size: int = 25_000
    max_chunks_per_file: int = 4
    max_scanned_rows_per_file: int = 100_000
    max_retained_rows_per_group_per_sensor: int = 20_000
    max_total_retained_rows: int = 80_000
    fixture_duration_seconds: float = 30.0


def _key(row: pd.Series) -> tuple[str, str, str]:
    return tuple(str(row[column]) for column in GROUP_COLUMNS)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def extract_fixture(
    data_directory: Path,
    output_directory: Path,
    groups: set[tuple[str, str, str]],
    limits: FixtureLimits = FixtureLimits(),
) -> dict[str, object]:
    """Extract one bounded fixture attempt; never reads beyond configured limits."""
    output_directory.mkdir(parents=True, exist_ok=True)
    retained: dict[str, dict[tuple[str, str, str], list[dict[str, object]]]] = {
        "acc": defaultdict(list),
        "gyro": defaultdict(list),
    }
    scan_report: list[dict[str, object]] = []
    total_retained = 0
    for sensor, filenames in SENSOR_FILES.items():
        for filename in filenames:
            path = data_directory / filename
            scanned = 0
            chunks = 0
            matching = 0
            retained_before = total_retained
            dtype = {
                "Index": "string",
                "Arrival_Time": "Int64",
                "Creation_Time": "Int64",
                "x": "float64",
                "y": "float64",
                "z": "float64",
                "User": "string",
                "Model": "string",
                "Device": "string",
                "gt": "string",
            }
            for chunk in pd.read_csv(
                path,
                usecols=list(CSV_COLUMNS),
                dtype=dtype,
                chunksize=limits.chunk_size,
                low_memory=False,
            ):
                if chunks >= limits.max_chunks_per_file:
                    break
                remaining = limits.max_scanned_rows_per_file - scanned
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()
                chunks += 1
                scanned += len(chunk)
                for row in chunk.to_dict("records"):
                    group = _key(pd.Series(row))
                    if group not in groups:
                        continue
                    matching += 1
                    if len(retained[sensor][group]) >= limits.max_retained_rows_per_group_per_sensor:
                        continue
                    if total_retained >= limits.max_total_retained_rows:
                        continue
                    retained[sensor][group].append(row)
                    total_retained += 1
                print(
                    f"{filename}: chunk={chunks} scanned={scanned} "
                    f"matching={matching} retained={total_retained}",
                    flush=True,
                )
                if scanned >= limits.max_scanned_rows_per_file:
                    break
            scan_report.append(
                {
                    "sensor": sensor,
                    "file": filename,
                    "chunks_scanned": chunks,
                    "rows_scanned": scanned,
                    "matching_rows": matching,
                    "rows_retained": total_retained - retained_before,
                    "scan_limit_reached": (
                        chunks >= limits.max_chunks_per_file
                        or scanned >= limits.max_scanned_rows_per_file
                    ),
                }
            )

    fixture_rows: dict[str, list[dict[str, object]]] = {"acc": [], "gyro": []}
    valid_groups: list[tuple[str, str, str]] = []
    incomplete_reasons: dict[str, int] = defaultdict(int)
    for group in sorted(groups):
        acc_rows = retained["acc"].get(group, [])
        gyro_rows = retained["gyro"].get(group, [])
        if not acc_rows or not gyro_rows:
            incomplete_reasons["missing_sensor_in_bounded_scan"] += 1
            continue
        acc_times = [int(row["Creation_Time"]) for row in acc_rows if pd.notna(row["Creation_Time"])]
        gyro_times = [int(row["Creation_Time"]) for row in gyro_rows if pd.notna(row["Creation_Time"])]
        if not acc_times or not gyro_times:
            incomplete_reasons["missing_valid_creation_time"] += 1
            continue
        start = max(min(acc_times), min(gyro_times))
        end = min(max(acc_times), max(gyro_times))
        if end <= start:
            incomplete_reasons["no_timestamp_intersection"] += 1
            continue
        bounded_end = min(end, start + int(limits.fixture_duration_seconds * 1_000_000_000))
        acc_selected = [
            row for row in acc_rows
            if start <= int(row["Creation_Time"]) <= bounded_end
        ]
        gyro_selected = [
            row for row in gyro_rows
            if start <= int(row["Creation_Time"]) <= bounded_end
        ]
        if len(acc_selected) < 2 or len(gyro_selected) < 2:
            incomplete_reasons["insufficient_bounded_overlap_rows"] += 1
            continue
        fixture_rows["acc"].extend(acc_selected)
        fixture_rows["gyro"].extend(gyro_selected)
        valid_groups.append(group)

    status = (
        "complete"
        if valid_groups and len(valid_groups) == len(groups)
        else "partial_fixture"
        if valid_groups
        else "incomplete_fixture"
    )
    _write_rows(output_directory / "accelerometer.csv", fixture_rows["acc"])
    _write_rows(output_directory / "gyroscope.csv", fixture_rows["gyro"])
    manifest = {
        "status": status,
        "groups_requested": [list(group) for group in sorted(groups)],
        "groups_with_bounded_paired_coverage": [list(group) for group in valid_groups],
        "rows_written": {sensor: len(rows) for sensor, rows in fixture_rows.items()},
        "limits": asdict(limits),
        "scan_report": scan_report,
        "incomplete_reasons": dict(incomplete_reasons),
        "preservation": (
            "Rows retain source order within each source file, original integer "
            "timestamps, labels, and User/Model/Device identities. Disconnected "
            "fragments are not stitched."
        ),
    }
    (output_directory / "fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("ignore_files/data/full"))
    parser.add_argument("--output-directory", type=Path, default=Path("code/har_smoke_fixture_v2"))
    args = parser.parse_args()
    groups = {
        ("a", "nexus4", "nexus4_1"),
        ("g", "s3", "s3_2"),
        ("i", "s3mini", "s3mini_2"),
    }
    print(json.dumps(extract_fixture(args.data_directory, args.output_directory, groups), indent=2))


if __name__ == "__main__":
    main()
