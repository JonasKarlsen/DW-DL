from pathlib import Path

import pandas as pd

from dw_dl.har_fixture import FixtureLimits, extract_fixture


def _write_sensor(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_fixture_limits_and_paired_interval(tmp_path: Path):
    columns = [
        "Index", "Arrival_Time", "Creation_Time", "x", "y", "z",
        "User", "Model", "Device", "gt",
    ]
    rows = [
        dict(zip(columns, [i, i, i * 1_000_000_000, 1, 2, 3, "a", "m", "d", "walk"]))
        for i in range(10)
    ]
    for name in (
        "Phones_accelerometer.csv",
        "Phones_gyroscope.csv",
        "Watch_accelerometer.csv",
        "Watch_gyroscope.csv",
    ):
        _write_sensor(tmp_path / name, rows)
    manifest = extract_fixture(
        tmp_path,
        tmp_path / "fixture",
        {("a", "m", "d")},
        FixtureLimits(
            chunk_size=4,
            max_chunks_per_file=1,
            max_scanned_rows_per_file=4,
            max_retained_rows_per_group_per_sensor=4,
            max_total_retained_rows=8,
            fixture_duration_seconds=30,
        ),
    )
    assert manifest["status"] == "complete"
    assert manifest["rows_written"] == {"acc": 4, "gyro": 4}
