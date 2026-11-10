"""Memory-conscious audit of HHAR accelerometer and gyroscope recordings.

This module intentionally stops before alignment, resampling, windowing, or
model training. Timestamp values remain Python integers while differences are
computed, and only the differences are scaled for rate diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


def process_rss_bytes() -> int:
    """Return current process RSS without adding a runtime dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        if not get_info(handle, ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _check_rss(max_rss_bytes: int, peak: int) -> int:
    current = process_rss_bytes()
    peak = max(peak, current)
    if current > max_rss_bytes:
        raise MemoryError(
            f"process RSS {current / 2**20:.1f} MiB exceeds "
            f"limit {max_rss_bytes / 2**20:.1f} MiB"
        )
    return peak


GROUP_COLUMNS = ("User", "Model", "Device")
LABEL_COLUMN = "gt"
SENSOR_COLUMNS = ("x", "y", "z")
TIMESTAMP_COLUMNS = ("Arrival_Time", "Creation_Time")
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
CANDIDATE_SCALES = {
    "Arrival_Time": (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0),
    "Creation_Time": (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0),
}


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q * 99)])


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass
class TimeStats:
    """Streaming timestamp and interval statistics for one group and clock."""

    scale_candidates: tuple[float, ...]
    previous: int | None = None
    first: int | None = None
    last: int | None = None
    interval_samples: list[int] = field(default_factory=list)
    interval_count: int = 0
    duplicates: int = 0
    reversals: int = 0
    large_gaps: dict[float, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    resets: int = 0
    segment_counts: dict[float, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    segment_first: dict[float, int | None] = field(
        default_factory=lambda: defaultdict(lambda: None)
    )
    segment_last: dict[float, int | None] = field(
        default_factory=lambda: defaultdict(lambda: None)
    )
    segment_rows: dict[float, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    segment_records: dict[
        float, list[tuple[int, int, int, int, dict[str, int]]]
    ] = field(
        default_factory=lambda: defaultdict(list)
    )
    segment_usable_rows: dict[float, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    segment_labels: dict[float, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def _start_segment(self, scale: float, value: int) -> None:
        if (
            self.segment_first[scale] is not None
            and len(self.segment_records[scale]) < 128
        ):
            self.segment_records[scale].append(
                (
                    self.segment_first[scale],
                    self.segment_last[scale],
                    self.segment_rows[scale],
                    self.segment_usable_rows[scale],
                    dict(self.segment_labels[scale]),
                )
            )
        self.segment_counts[scale] += 1
        self.segment_first[scale] = value
        self.segment_last[scale] = value
        self.segment_rows[scale] = 0
        self.segment_usable_rows[scale] = 0
        self.segment_labels[scale].clear()

    def add(
        self,
        value: int,
        gap_seconds: float,
        usable: bool,
        label: str,
    ) -> None:
        if self.first is None:
            self.first = value
        for scale in self.scale_candidates:
            if self.segment_first[scale] is None:
                self._start_segment(scale, value)
        if self.previous is not None:
            delta = value - self.previous
            if delta == 0:
                self.duplicates += 1
            elif delta < 0:
                self.reversals += 1
                self.resets += 1
            else:
                self.interval_count += 1
                if len(self.interval_samples) < 4096:
                    self.interval_samples.append(delta)
                for scale in self.scale_candidates:
                    if delta / scale > gap_seconds:
                        self.large_gaps[scale] += 1
            for scale in self.scale_candidates:
                if delta <= 0 or delta / scale > gap_seconds:
                    self._start_segment(scale, value)
        for scale in self.scale_candidates:
            self.segment_rows[scale] += 1
            self.segment_last[scale] = value
            if usable:
                self.segment_usable_rows[scale] += 1
            if label:
                self.segment_labels[scale][label] += 1
        self.previous = value
        self.last = value

    def finalized_segments(
        self,
    ) -> dict[float, list[tuple[int, int, int, int, dict[str, int]]]]:
        result = {scale: list(records) for scale, records in self.segment_records.items()}
        for scale in self.scale_candidates:
            if self.segment_first[scale] is not None:
                result.setdefault(scale, []).append(
                    (
                        self.segment_first[scale],
                        self.segment_last[scale],
                        self.segment_rows[scale],
                        self.segment_usable_rows[scale],
                        dict(self.segment_labels[scale]),
                    )
                )
        return result

    def values(self, prefix: str) -> dict[str, object]:
        result: dict[str, object] = {
            f"{prefix}_first_raw": self.first,
            f"{prefix}_last_raw": self.last,
            f"{prefix}_interval_count": self.interval_count,
            f"{prefix}_duplicate_count": self.duplicates,
            f"{prefix}_reversal_count": self.reversals,
            f"{prefix}_reset_count": self.resets,
        }
        for scale in self.scale_candidates:
            name = f"{prefix}_scale_{scale:g}"
            seconds = [value / scale for value in self.interval_samples]
            result[f"{name}_median_hz"] = (
                1.0 / statistics.median(seconds) if seconds else None
            )
            result[f"{name}_interval_q01_s"] = _quantile(seconds, 0.01)
            result[f"{name}_interval_q50_s"] = _quantile(seconds, 0.50)
            result[f"{name}_interval_q99_s"] = _quantile(seconds, 0.99)
            result[f"{name}_effective_hz"] = (
                self.interval_count
                / ((self.last - self.first) / scale)
                if self.first is not None
                and self.last is not None
                and self.last > self.first
                else None
            )
            result[f"{name}_duration_s"] = (
                (self.last - self.first) / scale
                if self.first is not None and self.last is not None
                else None
            )
            result[f"{name}_large_gap_count"] = self.large_gaps[scale]
            result[f"{name}_segment_count"] = self.segment_counts[scale]
            result[f"{name}_segment_first_raw"] = self.segment_first[scale]
            result[f"{name}_segment_last_raw"] = self.segment_last[scale]
            result[f"{name}_segment_rows_last"] = self.segment_rows[scale]
        return result


@dataclass
class GroupState:
    """Streaming state for one participant/device/sensor combination."""

    user: str
    model: str
    device: str
    sensor: str
    rows: int = 0
    usable_rows: int = 0
    missing_timestamp: Counter[str] = field(default_factory=Counter)
    invalid_value_rows: int = 0
    labels: Counter[str] = field(default_factory=Counter)
    label_transitions: int = 0
    previous_label: str | None = None
    clocks: dict[str, TimeStats] = field(
        default_factory=lambda: {
            column: TimeStats(CANDIDATE_SCALES[column])
            for column in TIMESTAMP_COLUMNS
        }
    )

    def add(self, row: dict[str, object], gap_seconds: float) -> None:
        self.rows += 1
        label = str(row[LABEL_COLUMN]) if pd.notna(row[LABEL_COLUMN]) else ""
        if label:
            self.labels[label] += 1
            if self.previous_label is not None and label != self.previous_label:
                self.label_transitions += 1
            self.previous_label = label

        valid_values = all(_finite(row[column]) for column in SENSOR_COLUMNS)
        if not valid_values:
            self.invalid_value_rows += 1
        timestamp_valid = True
        for column in TIMESTAMP_COLUMNS:
            value = row[column]
            if pd.isna(value):
                self.missing_timestamp[column] += 1
                timestamp_valid = False
                continue
            try:
                integer_value = int(value)
            except (TypeError, ValueError):
                self.missing_timestamp[column] += 1
                timestamp_valid = False
                continue
            self.clocks[column].add(
                integer_value,
                gap_seconds,
                valid_values,
                label,
            )
        if timestamp_valid and valid_values:
            self.usable_rows += 1

    def values(self) -> dict[str, object]:
        result: dict[str, object] = {
            "User": self.user,
            "Model": self.model,
            "Device": self.device,
            "Sensor": self.sensor,
            "rows": self.rows,
            "usable_rows": self.usable_rows,
            "invalid_value_rows": self.invalid_value_rows,
            "missing_arrival_timestamps": self.missing_timestamp["Arrival_Time"],
            "missing_creation_timestamps": self.missing_timestamp["Creation_Time"],
            "label_count": len(self.labels),
            "label_counts": json.dumps(dict(sorted(self.labels.items()))),
            "label_transitions": self.label_transitions,
        }
        for column, clock in self.clocks.items():
            result.update(clock.values(column.lower()))
        return result


def _read_groups(
    path: Path,
    sensor: str,
    chunk_size: int,
    gap_seconds: float,
    max_rss_bytes: int,
    progress_handle: object,
    telemetry: dict[str, int],
) -> dict[tuple[str, str, str], GroupState]:
    states: dict[tuple[str, str, str], GroupState] = {}
    usecols = list(CSV_COLUMNS)
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
        usecols=usecols,
        dtype=dtype,
        chunksize=chunk_size,
        low_memory=False,
    ):
        telemetry["chunks"] += 1
        telemetry["rows_scanned"] += len(chunk)
        telemetry["peak_rss_bytes"] = _check_rss(
            max_rss_bytes, telemetry["peak_rss_bytes"]
        )
        progress_handle.write(
            json.dumps(
                {
                    "file": str(path),
                    "sensor": sensor,
                    "chunks": telemetry["chunks"],
                    "rows_scanned": telemetry["rows_scanned"],
                    "rss_bytes": process_rss_bytes(),
                }
            )
            + "\n"
        )
        progress_handle.flush()
        for values in chunk.itertuples(index=False, name=None):
            row = dict(zip(usecols, values))
            key = tuple(str(row[column]) for column in GROUP_COLUMNS)
            state = states.setdefault(
                key,
                GroupState(*key, sensor=sensor),
            )
            state.add(row, gap_seconds)
    return states


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pairing_rows(
    states: dict[str, dict[tuple[str, str, str], GroupState]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = sorted(set(states["acc"]) | set(states["gyro"]))
    for key in keys:
        acc = states["acc"].get(key)
        gyro = states["gyro"].get(key)
        base = dict(zip(GROUP_COLUMNS, key))
        base.update(
            {
                "accel_exists": acc is not None,
                "gyro_exists": gyro is not None,
                "pairing_status": (
                    "paired" if acc is not None and gyro is not None else "missing_sensor"
                ),
            }
        )
        for column in TIMESTAMP_COLUMNS:
            for scale in CANDIDATE_SCALES[column]:
                suffix = f"{column.lower()}_scale_{scale:g}"
                if acc is None or gyro is None:
                    metrics = {
                        "coverage_overlap_s": None,
                        "accel_duration_s": None,
                        "gyro_duration_s": None,
                        "duration_ratio": None,
                        "first_timestamp_offset_s": None,
                    }
                else:
                    aclock = acc.clocks[column]
                    gclock = gyro.clocks[column]
                    duration_a = (
                        (aclock.last - aclock.first) / scale
                        if aclock.first is not None and aclock.last is not None
                        else None
                    )
                    duration_g = (
                        (gclock.last - gclock.first) / scale
                        if gclock.first is not None and gclock.last is not None
                        else None
                    )
                    overlap = (
                        max(
                            0.0,
                            (
                                min(aclock.last, gclock.last)
                                - max(aclock.first, gclock.first)
                            )
                            / scale,
                        )
                        if None not in (aclock.first, aclock.last, gclock.first, gclock.last)
                        else None
                    )
                    metrics = {
                        "coverage_overlap_s": overlap,
                        "accel_duration_s": duration_a,
                        "gyro_duration_s": duration_g,
                        "duration_ratio": (
                            min(duration_a, duration_g) / max(duration_a, duration_g)
                            if duration_a and duration_g
                            else None
                        ),
                        "first_timestamp_offset_s": (
                            (gclock.first - aclock.first) / scale
                            if aclock.first is not None and gclock.first is not None
                            else None
                        ),
                        "median_rate_ratio_gyro_over_accel": (
                            (
                                statistics.median(
                                    [value / scale for value in aclock.interval_samples]
                                )
                                / statistics.median(
                                    [value / scale for value in gclock.interval_samples]
                                )
                            )
                            if aclock.interval_samples and gclock.interval_samples
                            else None
                        ),
                    }
                row = dict(base)
                row.update(
                    {
                        "timestamp_column": column,
                        "candidate_units_per_second": scale,
                        **metrics,
                        "interpretation_note": (
                            "Range overlap is not synchronization proof; compare "
                            "intervals, offsets, drift, and reset evidence."
                        ),
                    }
                )
                rows.append(row)
    return rows


def _segment_rows(
    states: dict[str, dict[tuple[str, str, str], GroupState]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sensor, sensor_states in states.items():
        for key in sorted(sensor_states):
            state = sensor_states[key]
            for column, clock in state.clocks.items():
                for scale, segments in clock.finalized_segments().items():
                    for segment_id, (
                        first,
                        last,
                        count,
                        usable_count,
                        labels,
                    ) in enumerate(segments):
                        rows.append(
                            {
                                **dict(zip(GROUP_COLUMNS, key)),
                                "Sensor": sensor,
                                "timestamp_column": column,
                                "candidate_units_per_second": scale,
                                "segment_id": segment_id,
                                "first_timestamp_raw": first,
                                "last_timestamp_raw": last,
                                "row_count": count,
                                "usable_row_count": usable_count,
                                "label_counts": json.dumps(
                                    dict(sorted(labels.items()))
                                ),
                                "duration_s": (last - first) / scale,
                                "candidate_rate_hz": (
                                    (count - 1) / ((last - first) / scale)
                                    if last > first
                                    else None
                                ),
                                "segment_definition": (
                                    "New segment after nonpositive timestamp "
                                    "difference or gap greater than the configured "
                                    "threshold under this candidate scale."
                                ),
                                "segment_rows_truncated": (
                                    state.clocks[column].segment_counts[scale] > 129
                                ),
                            }
                        )
    return rows


def _rate_rows(
    states: dict[str, dict[tuple[str, str, str], GroupState]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(set(states["acc"]) & set(states["gyro"])):
        acc = states["acc"][key]
        gyro = states["gyro"][key]
        for column in TIMESTAMP_COLUMNS:
            for scale in CANDIDATE_SCALES[column]:
                for target_hz in (25, 50):
                    a = acc.clocks[column]
                    g = gyro.clocks[column]
                    a_rate = a.values("tmp")[f"tmp_scale_{scale:g}_median_hz"]
                    g_rate = g.values("tmp")[f"tmp_scale_{scale:g}_median_hz"]
                    a_duration = (
                        (a.last - a.first) / scale
                        if a.first is not None and a.last is not None
                        else 0.0
                    )
                    g_duration = (
                        (g.last - g.first) / scale
                        if g.first is not None and g.last is not None
                        else 0.0
                    )
                    rows.append(
                        {
                            **dict(zip(GROUP_COLUMNS, key)),
                            "timestamp_column": column,
                            "candidate_units_per_second": scale,
                            "target_hz": target_hz,
                            "accel_typical_hz": a_rate,
                            "gyro_typical_hz": g_rate,
                            "accel_duration_s": a_duration,
                            "gyro_duration_s": g_duration,
                            "paired_duration_lower_bound_s": min(
                                a_duration, g_duration
                            ),
                            "accel_near_target": (
                                a_rate is not None and abs(a_rate - target_hz) <= target_hz * 0.2
                            ),
                            "gyro_near_target": (
                                g_rate is not None and abs(g_rate - target_hz) <= target_hz * 0.2
                            ),
                            "accel_supports_target_downsampling": (
                                a_rate is not None and a_rate >= target_hz
                            ),
                            "gyro_supports_target_downsampling": (
                                g_rate is not None and g_rate >= target_hz
                            ),
                            "paired_supports_target_downsampling": (
                                a_rate is not None
                                and g_rate is not None
                                and min(a_rate, g_rate) >= target_hz
                            ),
                            "activity_labels": json.dumps(
                                sorted(set(acc.labels) | set(gyro.labels))
                            ),
                        }
                    )
    return rows


def run_audit(
    data_directory: Path,
    output_directory: Path,
    chunk_size: int = 250_000,
    gap_seconds: float = 5.0,
    max_rss_mib: int = 1024,
) -> dict[str, object]:
    """Run the Stage 1 audit and write reports to a new directory."""
    output_directory.mkdir(parents=True, exist_ok=True)
    if max_rss_mib < 1:
        raise ValueError("max_rss_mib must be positive")
    max_rss_bytes = max_rss_mib * 2**20
    progress_path = output_directory / "progress.jsonl"
    telemetry = {"chunks": 0, "rows_scanned": 0, "peak_rss_bytes": 0}
    input_files = {
        "acc": {
            "Phones": data_directory / "Phones_accelerometer.csv",
            "Watch": data_directory / "Watch_accelerometer.csv",
        },
        "gyro": {
            "Phones": data_directory / "Phones_gyroscope.csv",
            "Watch": data_directory / "Watch_gyroscope.csv",
        },
    }
    states: dict[str, dict[tuple[str, str, str], GroupState]] = {
        "acc": {},
        "gyro": {},
    }
    with progress_path.open("w", encoding="utf-8") as progress_handle:
        for sensor, files in input_files.items():
            for path in files.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
                states[sensor].update(
                    _read_groups(
                        path,
                        sensor,
                        chunk_size,
                        gap_seconds,
                        max_rss_bytes,
                        progress_handle,
                        telemetry,
                    )
                )
                telemetry["peak_rss_bytes"] = _check_rss(
                    max_rss_bytes, telemetry["peak_rss_bytes"]
                )

    stream_rows = [
        state.values()
        for sensor_states in states.values()
        for state in sensor_states.values()
    ]
    _write_rows(output_directory / "stream_group_quality.csv", stream_rows)
    _write_rows(
        output_directory / "stream_segments.csv",
        _segment_rows(states),
    )
    _write_rows(
        output_directory / "paired_clock_compatibility.csv",
        _pairing_rows(states),
    )
    _write_rows(
        output_directory / "sampling_rate_comparison.csv",
        _rate_rows(states),
    )

    summary = {
        "audit_stage": 1,
        "data_directory": str(data_directory),
        "output_directory": str(output_directory),
        "chunk_size": chunk_size,
        "large_gap_threshold_seconds": gap_seconds,
        "group_keys": list(GROUP_COLUMNS),
        "timestamp_columns": list(TIMESTAMP_COLUMNS),
        "candidate_units_per_second": {
            column: list(scales) for column, scales in CANDIDATE_SCALES.items()
        },
        "timestamp_policy": (
            "Creation_Time and Arrival_Time are retained as nullable integers. "
            "Candidate scales are diagnostic only; no clock correction or "
            "independent zeroing is performed."
        ),
        "files": {
            sensor: {kind: str(path) for kind, path in files.items()}
            for sensor, files in input_files.items()
        },
        "stream_group_count": {
            sensor: len(sensor_states) for sensor, sensor_states in states.items()
        },
        "paired_group_count": len(set(states["acc"]) & set(states["gyro"])),
        "missing_accel_groups": len(set(states["gyro"]) - set(states["acc"])),
        "missing_gyro_groups": len(set(states["acc"]) - set(states["gyro"])),
        "status": "audit_complete_no_preprocessing_or_training_run",
    }
    (output_directory / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    summary.update(
        {
            "rows_scanned": telemetry["rows_scanned"],
            "peak_rss_bytes": telemetry["peak_rss_bytes"],
            "max_rss_mib": max_rss_mib,
            "shards_written": 0,
            "status": "audit_complete_no_preprocessing_or_training_run",
        }
    )
    (output_directory / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    artifacts = [
        "stream_group_quality.csv",
        "stream_segments.csv",
        "paired_clock_compatibility.csv",
        "sampling_rate_comparison.csv",
        "audit_summary.json",
        "progress.jsonl",
    ]
    manifest = {
        "artifacts": [
            {
                "filename": filename,
                "sha256": hashlib.sha256(
                    (output_directory / filename).read_bytes()
                ).hexdigest(),
            }
            for filename in artifacts
        ]
    }
    (output_directory / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_directory / "checkpoint.json").write_text(
        json.dumps(
            {
                "status": "committed",
                "artifact_manifest": "audit_manifest.json",
                "rows_scanned": telemetry["rows_scanned"],
                "peak_rss_bytes": telemetry["peak_rss_bytes"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("ignore_files/data/full"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("code/har_audit_v1"),
    )
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--gap-seconds", type=float, default=5.0)
    parser.add_argument("--max-rss-mib", type=int, default=1024)
    args = parser.parse_args()
    summary = run_audit(
        args.data_directory,
        args.output_directory,
        chunk_size=args.chunk_size,
        gap_seconds=args.gap_seconds,
        max_rss_mib=args.max_rss_mib,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
