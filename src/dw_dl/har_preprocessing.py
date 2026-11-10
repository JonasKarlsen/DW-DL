"""Raw HHAR alignment and windowing primitives for Stage 2.

The public functions are intentionally usable on small synthetic data and on
filtered CSV subsets. They do not perform a full-dataset run by themselves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from scipy import signal


GROUP_COLUMNS = ("User", "Model", "Device")
CHANNEL_ORDER = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
SENSOR_AXES = ("x", "y", "z")
LABEL_COLUMN = "gt"
TIMESTAMP_COLUMN = "Creation_Time"
TIMESTAMP_SCALE = 1_000_000_000


@dataclass(frozen=True)
class PreprocessingConfig:
    target_hz: float = 50.0
    window_seconds: float = 2.56
    overlap: float = 0.5
    session_gap_seconds: float = 5.0
    interpolation_gap_seconds: float = 0.25
    minimum_segment_overlap_seconds: float = 2.56
    lowpass_fraction_of_target: float = 0.4
    chunk_size: int = 50_000
    seed: int = 4786

    @property
    def samples_per_window(self) -> int:
        samples = self.window_seconds * self.target_hz
        if not math.isclose(samples, round(samples)):
            raise ValueError("window_seconds * target_hz must be an integer")
        return int(round(samples))

    @property
    def step_seconds(self) -> float:
        return self.window_seconds * (1.0 - self.overlap)

    @property
    def step_samples(self) -> int:
        samples = self.step_seconds * self.target_hz
        if not math.isclose(samples, round(samples)):
            raise ValueError("window step must be an integer number of samples")
        return int(round(samples))


@dataclass
class SensorSegment:
    timestamp: np.ndarray
    values: np.ndarray
    labels: np.ndarray
    segment_id: int
    start_raw: int
    end_raw: int
    source_positions: np.ndarray | None = None


@dataclass
class Rejection:
    reason: str
    user: str
    model: str
    device: str
    sensor_segment: str = ""
    start_raw: int | None = None
    end_raw: int | None = None


@dataclass(frozen=True)
class ReorderedBlock:
    sensor: str
    original_positions: tuple[int, int]
    original_indices: tuple[str, str]
    creation_times: tuple[int, int]
    original_rows: tuple[dict[str, object], dict[str, object]]
    backward_displacement_ns: int
    evidence: str


def _json_row(frame: pd.DataFrame, position: int) -> dict[str, object]:
    row: dict[str, object] = {}
    for column, value in frame.iloc[position].items():
        if isinstance(value, np.generic):
            value = value.item()
        if pd.isna(value):
            value = None
        row[str(column)] = value
    return row


def repair_adjacent_swaps(
    frame: pd.DataFrame,
    sensor: str,
    maximum_backward_ms: float = 20.0,
) -> tuple[pd.DataFrame, list[ReorderedBlock]]:
    """Repair only corroborated two-row adjacent swaps; never sort globally."""
    required = {"Index", "Arrival_Time", TIMESTAMP_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    repaired = frame.copy(deep=True)
    creation = pd.to_numeric(repaired[TIMESTAMP_COLUMN], errors="coerce")
    arrival = pd.to_numeric(repaired["Arrival_Time"], errors="coerce")
    index = pd.to_numeric(repaired["Index"], errors="coerce")
    records: list[ReorderedBlock] = []
    position = 0
    max_backward_ns = int(maximum_backward_ms * 1_000_000)
    while position + 1 < len(repaired):
        first = position
        second = position + 1
        if any(pd.isna(series.iloc[first]) or pd.isna(series.iloc[second])
               for series in (creation, arrival, index)):
            position += 1
            continue
        backward = int(creation.iloc[first] - creation.iloc[second])
        corroborated = (
            backward > 0
            and backward <= max_backward_ns
            and int(index.iloc[second]) == int(index.iloc[first]) - 1
            and int(arrival.iloc[first]) == int(arrival.iloc[second])
        )
        if not corroborated:
            position += 1
            continue
        before_ok = first == 0 or int(creation.iloc[first - 1]) < int(creation.iloc[second])
        after_ok = (
            second == len(repaired) - 1
            or int(creation.iloc[first]) < int(creation.iloc[second + 1])
        )
        if not (before_ok and after_ok):
            position += 1
            continue
        original_indices = (
            str(index.iloc[first]),
            str(index.iloc[second]),
        )
        original_creation_times = (
            int(creation.iloc[first]),
            int(creation.iloc[second]),
        )
        original_rows = (
            _json_row(repaired, first),
            _json_row(repaired, second),
        )
        rows = repaired.iloc[[first, second]].copy()
        repaired.iloc[[first, second]] = rows.iloc[[1, 0]].to_numpy()
        records.append(
            ReorderedBlock(
                sensor=sensor,
                original_positions=(first, second),
                original_indices=original_indices,
                creation_times=original_creation_times,
                original_rows=original_rows,
                backward_displacement_ns=backward,
                evidence="Index decrement by one, equal Arrival_Time, bounded backward Creation_Time, local swap restores order",
            )
        )
        creation = pd.to_numeric(repaired[TIMESTAMP_COLUMN], errors="coerce")
        arrival = pd.to_numeric(repaired["Arrival_Time"], errors="coerce")
        index = pd.to_numeric(repaired["Index"], errors="coerce")
        position += 2
    return repaired, records


def _finite_rows(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame[list(SENSOR_AXES)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
        & frame[TIMESTAMP_COLUMN].notna().to_numpy()
    )


def split_sensor_segments(
    frame: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[list[SensorSegment], list[Rejection]]:
    """Split one sensor/group in input order without sorting across resets."""
    required = {TIMESTAMP_COLUMN, *SENSOR_AXES, LABEL_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = frame.reset_index(drop=True)
    timestamps = pd.to_numeric(frame[TIMESTAMP_COLUMN], errors="coerce").astype("Int64")
    values = frame[list(SENSOR_AXES)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    labels = frame[LABEL_COLUMN].astype("string").fillna("").to_numpy(dtype=object)
    usable = _finite_rows(frame)
    segments: list[SensorSegment] = []
    rejections: list[Rejection] = []
    start = None
    previous: int | None = None
    segment_id = 0

    def finish(end: int) -> None:
        nonlocal start, segment_id
        if start is None:
            return
        indices = np.arange(start, end)
        valid = usable[indices]
        if valid.any():
            valid_indices = indices[valid]
            ts = np.array([int(timestamps.iloc[i]) for i in valid_indices], dtype=np.int64)
            order = np.argsort(ts, kind="stable")
            ts = ts[order]
            vals = values[valid_indices][order]
            labs = labels[valid_indices][order]
            positions = valid_indices[order]
            unique = np.r_[True, np.diff(ts) > 0]
            ts, vals, labs, positions = (
                ts[unique],
                vals[unique],
                labs[unique],
                positions[unique],
            )
            if len(ts) >= 2:
                segments.append(
                    SensorSegment(
                        ts,
                        vals,
                        labs,
                        segment_id,
                        int(ts[0]),
                        int(ts[-1]),
                        positions,
                    )
                )
                segment_id += 1
        start = None

    for index in range(len(frame)):
        raw = timestamps.iloc[index]
        if pd.isna(raw):
            if start is not None:
                finish(index)
            rejections.append(Rejection("missing_timestamp", "", "", ""))
            previous = None
            continue
        current = int(raw)
        if start is None:
            start = index
        elif previous is not None:
            delta = (current - previous) / TIMESTAMP_SCALE
            if delta <= 0:
                finish(index)
                start = index
                rejections.append(Rejection("clock_reset_or_out_of_order", "", "", ""))
            elif delta > config.session_gap_seconds:
                finish(index)
                start = index
                rejections.append(Rejection("session_gap", "", "", ""))
        previous = current
    finish(len(frame))
    return segments, rejections


def _label_at(
    source_timestamps: np.ndarray,
    labels: np.ndarray,
    target_timestamps: np.ndarray,
    max_gap_seconds: float,
) -> np.ndarray:
    positions = np.searchsorted(source_timestamps, target_timestamps)
    left = np.clip(positions - 1, 0, len(source_timestamps) - 1)
    right = np.clip(positions, 0, len(source_timestamps) - 1)
    choose_right = np.abs(source_timestamps[right] - target_timestamps) < np.abs(
        target_timestamps - source_timestamps[left]
    )
    nearest = np.where(choose_right, right, left)
    distances = np.abs(source_timestamps[nearest] - target_timestamps) / TIMESTAMP_SCALE
    result = labels[nearest].astype(object, copy=True)
    result[distances > max_gap_seconds] = ""
    return result


def _resample_values(
    source_timestamps: np.ndarray,
    values: np.ndarray,
    target_timestamps: np.ndarray,
    target_hz: float,
    lowpass_fraction: float,
) -> np.ndarray:
    source_seconds = (source_timestamps - source_timestamps[0]) / TIMESTAMP_SCALE
    target_seconds = (target_timestamps - source_timestamps[0]) / TIMESTAMP_SCALE
    intervals = np.diff(source_seconds)
    median_interval = float(np.median(intervals))
    source_hz = 1.0 / median_interval
    if source_hz < target_hz:
        raise ValueError("source_rate_below_target")
    uniform_step = median_interval
    uniform = np.arange(source_seconds[0], source_seconds[-1] + uniform_step / 2, uniform_step)
    regular = np.column_stack(
        [np.interp(uniform, source_seconds, values[:, axis]) for axis in range(values.shape[1])]
    )
    cutoff = target_hz * lowpass_fraction
    if cutoff >= source_hz / 2:
        filtered = regular
    else:
        sos = signal.butter(6, cutoff, btype="lowpass", fs=source_hz, output="sos")
        filtered = signal.sosfiltfilt(sos, regular, axis=0, padtype="odd")
    return np.column_stack(
        [np.interp(target_seconds, uniform, filtered[:, axis]) for axis in range(values.shape[1])]
    )


def align_sensor_segments(
    acc: SensorSegment,
    gyro: SensorSegment,
    config: PreprocessingConfig,
    key: tuple[str, str, str] = ("", "", ""),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Rejection]]:
    """Align a same-device segment pair on a shared grid without time shifting."""
    start = max(acc.timestamp[0], gyro.timestamp[0])
    end = min(acc.timestamp[-1], gyro.timestamp[-1])
    rejection: list[Rejection] = []
    if (end - start) / TIMESTAMP_SCALE < config.minimum_segment_overlap_seconds:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 6), dtype=np.float32),
            np.empty((0,), dtype=object),
            [Rejection("insufficient_intersection", *key)],
        )
    step_raw = int(round(TIMESTAMP_SCALE / config.target_hz))
    target = np.arange(start, end + 1, step_raw, dtype=np.int64)
    if len(target) < config.samples_per_window:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 6), dtype=np.float32),
            np.empty((0,), dtype=object),
            [Rejection("insufficient_intersection", *key)],
        )
    for segment in (acc, gyro):
        if np.any(
            np.diff(segment.timestamp) / TIMESTAMP_SCALE
            > config.interpolation_gap_seconds
        ):
            rejection.append(Rejection("interpolation_gap", *key, str(segment.segment_id)))
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 6), dtype=np.float32),
                np.empty((0,), dtype=object),
                rejection,
            )
    try:
        acc_values = _resample_values(
            acc.timestamp, acc.values, target, config.target_hz, config.lowpass_fraction_of_target
        )
        gyro_values = _resample_values(
            gyro.timestamp, gyro.values, target, config.target_hz, config.lowpass_fraction_of_target
        )
    except ValueError as exc:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 6), dtype=np.float32),
            np.empty((0,), dtype=object),
            [Rejection(str(exc), *key)],
        )
    acc_labels = _label_at(acc.timestamp, acc.labels, target, config.interpolation_gap_seconds)
    gyro_labels = _label_at(gyro.timestamp, gyro.labels, target, config.interpolation_gap_seconds)
    labels = np.where(acc_labels == gyro_labels, acc_labels, "")
    missing = (acc_labels == "") | (gyro_labels == "")
    disagreement = (acc_labels != gyro_labels) & ~missing
    if np.any(disagreement):
        rejection.append(Rejection("sensor_label_disagreement", *key))
    labels[missing | disagreement] = ""
    return target, np.column_stack((acc_values, gyro_values)).astype(np.float32), labels, rejection


def split_at_interpolation_gaps(
    segment: SensorSegment,
    config: PreprocessingConfig,
) -> list[SensorSegment]:
    """Split a session segment so interpolation never bridges a large gap."""
    boundaries = np.flatnonzero(
        np.diff(segment.timestamp) / TIMESTAMP_SCALE
        > config.interpolation_gap_seconds
    ) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(segment.timestamp)]
    result: list[SensorSegment] = []
    for sub_id, (start, end) in enumerate(zip(starts, ends)):
        if end - start < 2:
            continue
        timestamps = segment.timestamp[start:end]
        result.append(
            SensorSegment(
                timestamps,
                segment.values[start:end],
                segment.labels[start:end],
                f"{segment.segment_id}.{sub_id}",
                int(timestamps[0]),
                int(timestamps[-1]),
                (
                    segment.source_positions[start:end]
                    if segment.source_positions is not None
                    else None
                ),
            )
        )
    return result


def generate_windows(
    timestamps: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    key: tuple[str, str, str],
    segment_id: str,
    config: PreprocessingConfig,
) -> tuple[list[np.ndarray], list[str], list[dict[str, object]], list[Rejection]]:
    windows: list[np.ndarray] = []
    window_labels: list[str] = []
    metadata: list[dict[str, object]] = []
    rejections: list[Rejection] = []
    n = config.samples_per_window
    step = config.step_samples
    for start in range(0, len(values) - n + 1, step):
        end = start + n
        window_labels_raw = labels[start:end]
        if not np.all(window_labels_raw != ""):
            rejections.append(Rejection("invalid_or_missing_label", *key, segment_id))
            continue
        unique = np.unique(window_labels_raw)
        if len(unique) != 1:
            rejections.append(Rejection("mixed_label", *key, segment_id))
            continue
        windows.append(values[start:end].copy())
        window_labels.append(str(unique[0]))
        metadata.append(
            {
                **dict(zip(GROUP_COLUMNS, key)),
                "segment_id": segment_id,
                "window_id": f"{key[0]}_{key[2]}_{segment_id}_{start}",
                "start_timestamp_raw": int(timestamps[start]),
                "end_timestamp_raw": int(timestamps[end - 1]),
                "label": str(unique[0]),
            }
        )
    return windows, window_labels, metadata, rejections


def make_subject_splits(
    users: Iterable[str],
    seed: int = 4786,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, str]:
    unique = sorted(set(users))
    if len(unique) < 3:
        raise ValueError("at least three participants are required")
    rng = np.random.default_rng(seed)
    shuffled = list(np.array(unique)[rng.permutation(len(unique))])
    train_count = max(1, int(round(len(unique) * train_fraction)))
    validation_count = max(1, int(round(len(unique) * validation_fraction)))
    if train_count + validation_count >= len(unique):
        validation_count = max(1, len(unique) - train_count - 1)
    return {
        user: ("train" if index < train_count else "val" if index < train_count + validation_count else "test")
        for index, user in enumerate(shuffled)
    }


def fit_normalization(windows: np.ndarray, labels: np.ndarray | None = None) -> dict[str, list[float]]:
    if windows.ndim != 3 or windows.shape[-1] != 6:
        raise ValueError("expected windows with shape (N, samples, 6)")
    mean = windows.mean(axis=(0, 1), dtype=np.float64)
    std = windows.std(axis=(0, 1), dtype=np.float64)
    std[std < 1e-8] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_normalization(windows: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    return ((windows - np.asarray(stats["mean"])) / np.asarray(stats["std"])).astype(np.float32)


def save_dataset(
    output_directory: Path,
    windows: list[np.ndarray],
    labels: list[str],
    metadata: list[dict[str, object]],
    split_by_user: dict[str, str],
    config: PreprocessingConfig,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    arrays = np.asarray(windows, dtype=np.float32)
    encoded = {label: index for index, label in enumerate(sorted(set(labels)))}
    numeric = np.asarray([encoded[label] for label in labels], dtype=np.int64)
    split = np.asarray([split_by_user[row["User"]] for row in metadata])
    train_stats = fit_normalization(arrays[split == "train"])
    np.save(output_directory / "X_raw.npy", arrays)
    np.save(output_directory / "y.npy", numeric)
    np.save(output_directory / "split.npy", split)
    np.save(output_directory / "X.npy", apply_normalization(arrays, train_stats))
    with (output_directory / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    (output_directory / "label_mapping.json").write_text(json.dumps(encoded, indent=2), encoding="utf-8")
    (output_directory / "normalization.json").write_text(json.dumps(train_stats, indent=2), encoding="utf-8")
    (output_directory / "configuration.json").write_text(
        json.dumps({**asdict(config), "channel_order": CHANNEL_ORDER}, indent=2),
        encoding="utf-8",
    )


def save_diagnostic_plots(
    output_directory: Path,
    timestamps: np.ndarray,
    aligned: np.ndarray,
    accepted_window: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    seconds = (timestamps - timestamps[0]) / TIMESTAMP_SCALE
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(seconds, aligned[:, 0], label="acc_x")
    axes[0].plot(seconds, aligned[:, 3], label="gyro_x")
    axes[0].set_ylabel("aligned value")
    axes[0].legend()
    axes[0].set_title("Smoke-test aligned sensor observations")
    if accepted_window is not None:
        axes[1].plot(accepted_window[:, 0], label="acc_x")
        axes[1].plot(accepted_window[:, 3], label="gyro_x")
        axes[1].set_title("Representative accepted raw window")
        axes[1].set_xlabel("sample")
        axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_directory / "alignment_and_window.png", dpi=150)
    plt.close(figure)


def save_fixture_failure_diagnostics(
    fixture_directory: Path,
    output_directory: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> dict[str, object]:
    """Plot saved-fixture timestamp anomalies and current fragment coverage."""
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    frames = {
        "acc": pd.read_csv(fixture_directory / "accelerometer.csv"),
        "gyro": pd.read_csv(fixture_directory / "gyroscope.csv"),
    }
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    fragment_counts: dict[str, int] = {}
    fragment_durations: dict[str, list[float]] = {}
    origins = [
        int(frame[TIMESTAMP_COLUMN].astype("int64").iloc[0])
        for frame in frames.values()
    ]
    for sensor, frame in frames.items():
        timestamps = frame[TIMESTAMP_COLUMN].astype("int64").to_numpy()
        origin = min(origins)
        axes[0].plot(
            np.arange(len(timestamps)),
            (timestamps - origin) / TIMESTAMP_SCALE,
            ".",
            markersize=2,
            label=sensor,
        )
        differences = np.diff(timestamps) / TIMESTAMP_SCALE
        axes[1].plot(
            np.arange(1, len(timestamps)),
            differences,
            ".",
            markersize=2,
            label=sensor,
        )
        sessions, _ = split_sensor_segments(frame, config)
        fragments = sum(
            (split_at_interpolation_gaps(session, config) for session in sessions),
            [],
        )
        fragment_counts[sensor] = len(fragments)
        fragment_durations[sensor] = [
            (fragment.timestamp[-1] - fragment.timestamp[0]) / TIMESTAMP_SCALE
            for fragment in fragments
        ]
        for fragment in fragments:
            axes[2].plot(
                [
                    fragment.timestamp[0] / TIMESTAMP_SCALE,
                    fragment.timestamp[-1] / TIMESTAMP_SCALE,
                ],
                [sensor, sensor],
                linewidth=3,
                alpha=0.7,
            )
    axes[0].set_title("Creation_Time relative to shared fixture origin")
    axes[0].set_ylabel("seconds")
    axes[0].legend()
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(config.interpolation_gap_seconds, color="red", linestyle="--")
    axes[1].set_title("Timestamp differences in source row order")
    axes[1].set_ylabel("seconds")
    axes[1].legend()
    axes[2].set_title("Current continuous-fragment coverage")
    axes[2].set_xlabel("Creation_Time seconds")
    axes[2].set_yticks(["acc", "gyro"])
    figure.savefig(output_directory / "fixture_failure_diagnostics.png", dpi=150)
    plt.close(figure)
    summary = {
        "fragment_counts": fragment_counts,
        "fragment_durations_seconds": fragment_durations,
        "plot": str(output_directory / "fixture_failure_diagnostics.png"),
    }
    (output_directory / "fragment_diagnostics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def iter_csv_groups(
    path: Path,
    groups: set[tuple[str, str, str]],
    chunk_size: int,
) -> Iterator[tuple[tuple[str, str, str], pd.DataFrame]]:
    usecols = ["Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"]
    dtype = {
        "Creation_Time": "Int64",
        "x": "float64",
        "y": "float64",
        "z": "float64",
        "User": "string",
        "Model": "string",
        "Device": "string",
        "gt": "string",
    }
    collected: dict[tuple[str, str, str], list[pd.DataFrame]] = {group: [] for group in groups}
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunk_size, low_memory=False):
        keys = list(zip(chunk.User.astype(str), chunk.Model.astype(str), chunk.Device.astype(str)))
        for group in groups:
            mask = np.fromiter((key == group for key in keys), dtype=bool, count=len(keys))
            if mask.any():
                collected[group].append(chunk.loc[mask].copy())
    for group, frames in collected.items():
        if frames:
            yield group, pd.concat(frames, ignore_index=True)


def preprocess_groups(
    data_directory: Path,
    groups: set[tuple[str, str, str]],
    output_directory: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> dict[str, object]:
    """Process only explicitly selected groups; intended for smoke tests."""
    streams: dict[str, dict[tuple[str, str, str], pd.DataFrame]] = {"acc": {}, "gyro": {}}
    for sensor, filename in (
        ("acc", "Phones_accelerometer.csv"),
        ("acc", "Watch_accelerometer.csv"),
        ("gyro", "Phones_gyroscope.csv"),
        ("gyro", "Watch_gyroscope.csv"),
    ):
        print(f"reading {filename} for {len(groups)} selected groups", flush=True)
        for group, frame in iter_csv_groups(data_directory / filename, groups, config.chunk_size):
            streams[sensor][group] = frame
    windows: list[np.ndarray] = []
    labels: list[str] = []
    metadata: list[dict[str, object]] = []
    rejections: list[Rejection] = []
    diagnostic_alignment: tuple[np.ndarray, np.ndarray] | None = None
    for group in sorted(groups):
        if group not in streams["acc"] or group not in streams["gyro"]:
            rejections.append(Rejection("missing_sensor", *group))
            continue
        acc_segments, acc_rejections = split_sensor_segments(streams["acc"][group], config)
        gyro_segments, gyro_rejections = split_sensor_segments(streams["gyro"][group], config)
        rejections.extend(acc_rejections + gyro_rejections)
        for acc_session in acc_segments:
            for gyro_session in gyro_segments:
                acc_parts = split_at_interpolation_gaps(acc_session, config)
                gyro_parts = split_at_interpolation_gaps(gyro_session, config)
                for acc_segment in acc_parts:
                    for gyro_segment in gyro_parts:
                        timestamps, values, aligned_labels, alignment_rejections = align_sensor_segments(
                            acc_segment, gyro_segment, config, group
                        )
                        rejections.extend(alignment_rejections)
                        if len(timestamps) == 0:
                            continue
                        if diagnostic_alignment is None:
                            diagnostic_alignment = (timestamps, values)
                        result = generate_windows(
                            timestamps,
                            values,
                            aligned_labels,
                            group,
                            f"{acc_segment.segment_id}_{gyro_segment.segment_id}",
                            config,
                        )
                        windows.extend(result[0])
                        labels.extend(result[1])
                        metadata.extend(result[2])
                        rejections.extend(result[3])
    if not windows:
        raise RuntimeError("smoke subset produced no accepted windows")
    split_by_user = make_subject_splits((row["User"] for row in metadata), config.seed)
    save_dataset(output_directory, windows, labels, metadata, split_by_user, config)
    if diagnostic_alignment is not None:
        save_diagnostic_plots(
            output_directory,
            diagnostic_alignment[0],
            diagnostic_alignment[1],
            windows[0],
        )
    rejection_counts = pd.Series([item.reason for item in rejections], dtype="string").value_counts().to_dict()
    (output_directory / "rejections.json").write_text(json.dumps(rejection_counts, indent=2), encoding="utf-8")
    return {
        "groups_requested": len(groups),
        "accepted_windows": len(windows),
        "window_shape": list(windows[0].shape),
        "rejections": rejection_counts,
        "output_directory": str(output_directory),
    }


def preprocess_saved_fixture(
    fixture_directory: Path,
    output_directory: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
) -> dict[str, object]:
    """Preprocess only the two raw CSVs already saved in a fixture directory."""
    streams = {
        "acc": pd.read_csv(fixture_directory / "accelerometer.csv"),
        "gyro": pd.read_csv(fixture_directory / "gyroscope.csv"),
    }
    repaired_blocks: list[ReorderedBlock] = []
    for sensor in ("acc", "gyro"):
        streams[sensor], records = repair_adjacent_swaps(streams[sensor], sensor)
        repaired_blocks.extend(records)
    group_columns = ["User", "Model", "Device"]
    groups = set(
        map(
            tuple,
            streams["acc"][group_columns]
            .drop_duplicates()
            .astype(str)
            .to_numpy(),
        )
    ) & set(
        map(
            tuple,
            streams["gyro"][group_columns]
            .drop_duplicates()
            .astype(str)
            .to_numpy(),
        )
    )
    windows: list[np.ndarray] = []
    labels: list[str] = []
    metadata: list[dict[str, object]] = []
    rejections: list[Rejection] = []
    diagnostic_alignment: tuple[np.ndarray, np.ndarray] | None = None
    for group in sorted(groups):
        group_frames = {
            sensor: streams[sensor][
                (streams[sensor]["User"].astype(str) == group[0])
                & (streams[sensor]["Model"].astype(str) == group[1])
                & (streams[sensor]["Device"].astype(str) == group[2])
            ].reset_index(drop=True)
            for sensor in ("acc", "gyro")
        }
        acc_sessions, acc_rejections = split_sensor_segments(group_frames["acc"], config)
        gyro_sessions, gyro_rejections = split_sensor_segments(group_frames["gyro"], config)
        rejections.extend(acc_rejections + gyro_rejections)
        for acc_session in acc_sessions:
            for gyro_session in gyro_sessions:
                for acc_segment in split_at_interpolation_gaps(acc_session, config):
                    for gyro_segment in split_at_interpolation_gaps(gyro_session, config):
                        timestamps, values, aligned_labels, alignment_rejections = align_sensor_segments(
                            acc_segment, gyro_segment, config, group
                        )
                        rejections.extend(alignment_rejections)
                        if len(timestamps) == 0:
                            continue
                        if diagnostic_alignment is None:
                            diagnostic_alignment = (timestamps, values)
                        result = generate_windows(
                            timestamps,
                            values,
                            aligned_labels,
                            group,
                            f"{acc_segment.segment_id}_{gyro_segment.segment_id}",
                            config,
                        )
                        windows.extend(result[0])
                        labels.extend(result[1])
                        metadata.extend(result[2])
    output_directory.mkdir(parents=True, exist_ok=True)
    rejection_counts = pd.Series(
        [item.reason for item in rejections], dtype="string"
    ).value_counts().to_dict()
    (output_directory / "rejections.json").write_text(
        json.dumps(rejection_counts, indent=2), encoding="utf-8"
    )
    (output_directory / "repaired_blocks.json").write_text(
        json.dumps([asdict(record) for record in repaired_blocks], indent=2),
        encoding="utf-8",
    )
    if windows:
        arrays = np.asarray(windows, dtype=np.float32)
        np.save(output_directory / "X_raw.npy", arrays)
        np.save(
            output_directory / "labels.npy",
            np.asarray(labels, dtype="<U32"),
        )
        with (output_directory / "metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)
        if diagnostic_alignment is not None:
            save_diagnostic_plots(
                output_directory,
                diagnostic_alignment[0],
                diagnostic_alignment[1],
                windows[0],
            )
    summary = {
        "status": "complete" if windows else "no_accepted_windows",
        "fixture_directory": str(fixture_directory),
        "output_directory": str(output_directory),
        "groups_processed": [list(group) for group in sorted(groups)],
        "accepted_windows": len(windows),
        "window_shape": list(windows[0].shape) if windows else [0, 0, 6],
        "target_hz": config.target_hz,
        "window_seconds": config.window_seconds,
        "overlap": config.overlap,
        "normalized": False,
        "rejections": rejection_counts,
        "repaired_block_count": len(repaired_blocks),
        "repaired_row_count": len(repaired_blocks) * 2,
        "repair_policy": {
            "maximum_block_size": 2,
            "maximum_backward_ms": 20.0,
            "required_evidence": [
                "adjacent Index decrement by one",
                "equal Arrival_Time",
                "local Creation_Time order restored by swap",
            ],
        },
    }
    (output_directory / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=Path("ignore_files/data/full"))
    parser.add_argument("--output-directory", type=Path, default=Path("code/har_smoke_v1"))
    parser.add_argument("--user", action="append", default=["a", "g", "i"])
    args = parser.parse_args()
    groups = {
        ("a", "nexus4", "nexus4_1"),
        ("g", "s3", "s3_2"),
        ("i", "s3mini", "s3mini_2"),
    }
    print(preprocess_groups(args.data_directory, groups, args.output_directory))


if __name__ == "__main__":
    main()
