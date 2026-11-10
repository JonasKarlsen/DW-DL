"""Bounded streaming primitives for fixture-sized preprocessing experiments."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from .har_preprocessing import (
    GROUP_COLUMNS,
    PreprocessingConfig,
    Rejection,
    SensorSegment,
    align_sensor_segments,
    generate_windows,
    repair_adjacent_swaps,
    split_at_interpolation_gaps,
    split_sensor_segments,
)
from .har_audit import process_rss_bytes

STAGE2_MAX_IN_MEMORY_FRAGMENT_SAMPLES = 500_000
STAGE2_RESAMPLING_BLOCK_SAMPLES = 25_000
STAGE2_MAX_DISK_BACKED_FRAGMENT_SAMPLES = 5_000_000
STAGE2_RSS_LIMIT_BYTES = 1_024 * 1024 * 1024


def _input_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _median_rate_hz(timestamps: np.ndarray) -> float | None:
    if len(timestamps) < 2:
        return None
    intervals = np.diff(timestamps) / 1_000_000_000
    if not np.all(intervals > 0):
        return None
    return float(1.0 / np.median(intervals))


def _stage2_configuration_id(
    config: PreprocessingConfig,
    stage2_config: BoundedStage2Config,
    chunk_size: int,
    groups: list[tuple[str, str, str]],
) -> str:
    payload = {
        "preprocessing": asdict(config),
        "stage2": asdict(stage2_config),
        "chunk_size": chunk_size,
        "groups": groups,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class BoundedStage2Config:
    max_in_memory_fragment_samples: int = STAGE2_MAX_IN_MEMORY_FRAGMENT_SAMPLES
    resampling_block_samples: int = STAGE2_RESAMPLING_BLOCK_SAMPLES
    max_disk_backed_fragment_samples: int = STAGE2_MAX_DISK_BACKED_FRAGMENT_SAMPLES
    rss_limit_bytes: int = STAGE2_RSS_LIMIT_BYTES
    windows_per_shard: int = 1_000


def _check_stage2_rss(config: BoundedStage2Config) -> None:
    if process_rss_bytes() > config.rss_limit_bytes:
        raise MemoryError("Stage 2 process RSS ceiling exceeded")


def interval_sweep_overlaps(
    left: list[SensorSegment], right: list[SensorSegment]
) -> list[tuple[SensorSegment, SensorSegment]]:
    """Return only overlapping intervals using a sorted active-set sweep."""
    left_sorted = sorted(left, key=lambda item: (item.start_raw, item.end_raw))
    right_sorted = sorted(right, key=lambda item: (item.start_raw, item.end_raw))
    active: list[SensorSegment] = []
    right_index = 0
    pairs: list[tuple[SensorSegment, SensorSegment]] = []
    for current in left_sorted:
        while right_index < len(right_sorted) and right_sorted[right_index].start_raw <= current.end_raw:
            active.append(right_sorted[right_index])
            right_index += 1
        active = [item for item in active if item.end_raw >= current.start_raw]
        pairs.extend((current, item) for item in active)
    return pairs


def _spool_repaired_stream(
    path: Path,
    sensor: str,
    spool_directory: Path,
    repair_handle,
    chunk_size: int,
    rss_config: BoundedStage2Config,
    selected_groups: set[tuple[str, str, str]] | None = None,
) -> set[tuple[str, str, str]]:
    state = ChunkRepairState()
    paths: dict[tuple[str, str, str], Path] = {}
    headers: dict[tuple[str, str, str], bool] = {}
    started_at = time.monotonic()
    chunks_read = 0
    rows_read = 0
    selected_rows = 0
    rows_spooled = 0

    def write_rows(rows: pd.DataFrame) -> int:
        written = 0
        for group, group_frame in rows.groupby(list(GROUP_COLUMNS), sort=False):
            key = tuple(str(value) for value in group)
            target = paths.setdefault(key, spool_directory / f"{sensor}-{len(paths):06d}.csv")
            group_frame.to_csv(
                target,
                mode="a",
                header=not headers.get(key, False),
                index=False,
            )
            headers[key] = True
            written += len(group_frame)
        return written

    reader = iter(pd.read_csv(path, chunksize=chunk_size))
    while True:
        _check_stage2_rss(rss_config)
        try:
            chunk = next(reader)
        except StopIteration:
            break
        chunks_read += 1
        rows_read += len(chunk)
        if selected_groups is not None:
            chunk = chunk[
                pd.MultiIndex.from_frame(
                    chunk[list(GROUP_COLUMNS)].astype(str)
                ).isin(selected_groups)
            ]
        selected_rows += len(chunk)
        rows, records = state.process_chunk(chunk, sensor)
        rows_spooled += write_rows(rows)
        if repair_handle is not None:
            for record in records:
                repair_handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        print(
            f"[stage2-spool] sensor={sensor} chunks={chunks_read} "
            f"rows_read={rows_read} elapsed_seconds={time.monotonic() - started_at:.1f} "
            f"selected_rows={selected_rows} rows_spooled={rows_spooled}",
            flush=True,
        )
    tail, records = state.finish(sensor)
    rows_spooled += write_rows(tail)
    if repair_handle is not None:
        for record in records:
            repair_handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    print(
        f"[stage2-spool] sensor={sensor} chunks={chunks_read} "
        f"rows_read={rows_read} elapsed_seconds={time.monotonic() - started_at:.1f} "
        f"selected_rows={selected_rows} rows_spooled={rows_spooled}",
        flush=True,
    )
    return set(paths)

REQUIRED_METADATA_FIELDS = frozenset(
    {
        "shard_sequence",
        "window_index",
        "user",
        "model",
        "device",
        "start_timestamp_raw",
        "end_timestamp_raw",
        "label",
        "config_version",
    }
)
TRACEABILITY_METADATA_FIELDS = frozenset(
    {"sensor_pair", "source_segment_ids", "source_row_positions", "source_traceability_ref"}
)


@dataclass
class ChunkRepairState:
    """Per-group look-behind state for the provisional adjacent-swap repair."""

    pending_rows: dict[tuple[str, str, str, str], list[pd.Series]] = field(default_factory=dict)
    pending_positions: dict[tuple[str, str, str, str], list[int]] = field(default_factory=dict)
    group_row_counts: dict[tuple[str, str, str], int] = field(default_factory=dict)
    last_emitted_rows: dict[tuple[str, str, str, str], pd.Series] = field(default_factory=dict)
    pending_eligible_pair: dict[tuple[str, str, str, str], bool] = field(default_factory=dict)
    group_context: dict[tuple[str, str, str], tuple[str, str]] = field(default_factory=dict)
    boundary_context: dict[tuple[str, str, str], str] = field(default_factory=dict)

    def process_chunk(
        self, chunk: pd.DataFrame, sensor: str
    ) -> tuple[pd.DataFrame, list[object]]:
        if not set(GROUP_COLUMNS).issubset(chunk.columns):
            raise ValueError(f"missing columns: {sorted(set(GROUP_COLUMNS) - set(chunk.columns))}")
        emitted: list[pd.Series] = []
        records: list[object] = []
        for _, row in chunk.iterrows():
            group_key = tuple(str(row[column]) for column in GROUP_COLUMNS)
            boundary = self._boundary_token(row)
            source_position = self.group_row_counts.get(group_key, 0)
            self.group_row_counts[group_key] = source_position + 1
            previous_boundary = self.boundary_context.get(group_key)
            if previous_boundary is not None and previous_boundary != boundary:
                old_key = (*group_key, previous_boundary)
                emitted.extend(self.pending_rows.pop(old_key, []))
                self.pending_positions.pop(old_key, None)
                self.pending_eligible_pair.pop(old_key, None)
            self.boundary_context[group_key] = boundary
            key = (*group_key, boundary)
            buffer = self.pending_rows.setdefault(key, [])
            positions = self.pending_positions.setdefault(key, [])
            buffer.append(row.copy())
            positions.append(source_position)
            self.group_context[group_key] = (sensor, boundary)
            if len(buffer) < 3:
                self._update_pending_flag(key)
                continue
            context = self.last_emitted_rows.get(key)
            candidate = pd.DataFrame(([context] if context is not None else []) + buffer)
            repaired, candidate_records = repair_adjacent_swaps(candidate, sensor)
            offset = 1 if context is not None else 0
            pair_record = next(
                (
                    record for record in candidate_records
                    if record.original_positions == (offset, offset + 1)
                ),
                None,
            )
            if pair_record is not None:
                emitted.extend(
                    repaired.iloc[index].copy() for index in (offset, offset + 1)
                )
                self.last_emitted_rows[key] = repaired.iloc[offset + 1].copy()
                records.append(
                    replace(
                        pair_record,
                        original_positions=(
                            positions[0],
                            positions[1],
                        ),
                    )
                )
                self.pending_rows[key] = buffer[2:]
                self.pending_positions[key] = positions[2:]
            else:
                emitted.append(buffer[0].copy())
                self.last_emitted_rows[key] = buffer[0].copy()
                self.pending_rows[key] = buffer[1:]
                self.pending_positions[key] = positions[1:]
            self._update_pending_flag(key)
        if not emitted:
            return chunk.iloc[0:0].copy(), records
        return pd.DataFrame(emitted, columns=chunk.columns), records

    def finish(self, sensor: str) -> tuple[pd.DataFrame, list[object]]:
        emitted: list[pd.Series] = []
        records: list[object] = []
        for key in list(self.pending_rows):
            buffer = self.pending_rows.pop(key)
            positions = self.pending_positions.pop(key)
            if buffer:
                context = self.last_emitted_rows.get(key)
                candidate = pd.DataFrame(([context] if context is not None else []) + buffer)
                repaired, candidate_records = repair_adjacent_swaps(candidate, sensor)
                offset = 1 if context is not None else 0
                emitted.extend(
                    repaired.iloc[index].copy()
                    for index in range(offset, len(repaired))
                )
                records.extend(
                    replace(
                        record,
                        original_positions=(
                            positions[record.original_positions[0] - offset],
                            positions[record.original_positions[1] - offset],
                        ),
                    )
                    for record in candidate_records
                    if record.original_positions[0] >= offset
                )
            self.pending_eligible_pair.pop(key, None)
            self.group_context.pop(key[:3], None)
            self.boundary_context.pop(key[:3], None)
            self.group_row_counts.pop(key[:3], None)
            self.last_emitted_rows.pop(key, None)
        if not emitted:
            return pd.DataFrame(), records
        return pd.DataFrame(emitted), records

    def _update_pending_flag(self, key: tuple[str, str, str, str]) -> None:
        rows = self.pending_rows[key]
        if len(rows) < 2:
            self.pending_eligible_pair[key] = False
            return
        candidate = pd.DataFrame(rows)
        _, records = repair_adjacent_swaps(candidate, "stream")
        self.pending_eligible_pair[key] = bool(records)

    @staticmethod
    def _boundary_token(row: pd.Series) -> str:
        boundary_columns = (
            "__boundary_id",
            "Session",
            "session_id",
            "Clock_Epoch",
            "clock_epoch",
            "segment_id",
        )
        for column in boundary_columns:
            if column in row.index:
                value = row[column]
                return f"{column}={value}"
        return "__single_boundary__"


@dataclass
class FilterState:
    """State marker that makes segment-boundary filter resets observable."""

    segment_id: str | None = None
    reset_count: int = 0

    def begin_segment(self, segment_id: str) -> None:
        if self.segment_id != segment_id:
            self.segment_id = segment_id
            self.reset_count += 1


@dataclass
class WindowOverlapBuffer:
    samples_per_window: int
    step_samples: int
    values: np.ndarray = field(default_factory=lambda: np.empty((0, 6), dtype=np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=object))

    def append(
        self, values: np.ndarray, labels: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(values) != len(labels):
            raise ValueError("values and labels must have equal lengths")
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError("values must have shape (n, 6)")
        self.values = np.concatenate((self.values, values), axis=0)
        self.labels = np.concatenate((self.labels, labels.astype(object)), axis=0)
        windows: list[tuple[np.ndarray, np.ndarray]] = []
        while len(self.values) >= self.samples_per_window:
            windows.append(
                (
                    self.values[: self.samples_per_window].copy(),
                    self.labels[: self.samples_per_window].copy(),
                )
            )
            self.values = self.values[self.step_samples :]
            self.labels = self.labels[self.step_samples :]
        return windows


@dataclass
class CompressedShardWriter:
    output_directory: Path
    windows_per_shard: int = 1_000
    _values: list[np.ndarray] = field(default_factory=list, init=False)
    _labels: list[str] = field(default_factory=list, init=False)
    _metadata: list[dict[str, object]] = field(default_factory=list, init=False)
    _sequence: int = field(default=0, init=False)
    _manifest: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.windows_per_shard < 1:
            raise ValueError("windows_per_shard must be positive")
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def append(
        self, values: np.ndarray, label: str, metadata: dict[str, object]
    ) -> None:
        missing = REQUIRED_METADATA_FIELDS - metadata.keys()
        if missing or not metadata.keys() & TRACEABILITY_METADATA_FIELDS:
            missing_text = sorted(missing)
            if not metadata.keys() & TRACEABILITY_METADATA_FIELDS:
                missing_text.append("one traceability field")
            raise ValueError(f"metadata missing required fields: {missing_text}")
        if metadata["shard_sequence"] != self._sequence:
            raise ValueError("metadata shard_sequence does not match current shard")
        if metadata["window_index"] != len(self._values):
            raise ValueError("metadata window_index does not match current shard position")
        self._values.append(np.asarray(values, dtype=np.float32))
        self._labels.append(str(label))
        self._metadata.append(dict(metadata))
        if len(self._values) >= self.windows_per_shard:
            self.flush()

    @property
    def next_sequence(self) -> int:
        return self._sequence

    @property
    def next_window_index(self) -> int:
        return len(self._values)

    def flush(self) -> None:
        if not self._values:
            return
        sequence = self._sequence
        filename = f"windows-{sequence:06d}.npz"
        temporary = self.output_directory / f".{filename}.tmp"
        target = self.output_directory / filename
        metadata_json = np.asarray(
            [json.dumps(item, sort_keys=True) for item in self._metadata],
            dtype="<U4096",
        )
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                X=np.asarray(self._values, dtype=np.float32),
                labels=np.asarray(self._labels, dtype="<U128"),
                metadata=metadata_json,
                shard_sequence=np.asarray(sequence, dtype=np.int64),
            )
        os.replace(temporary, target)
        checksum = hashlib.sha256(target.read_bytes()).hexdigest()
        self._manifest.append(
            {
                "sequence": sequence,
                "filename": filename,
                "window_count": len(self._values),
                "sha256": checksum,
            }
        )
        self._write_manifest()
        self._sequence += 1
        self._values.clear()
        self._labels.clear()
        self._metadata.clear()

    def close(self) -> None:
        self.flush()

    def _write_manifest(self) -> None:
        temporary = self.output_directory / ".shard_manifest.json.tmp"
        temporary.write_text(
            json.dumps({"shards": self._manifest}, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.output_directory / "shard_manifest.json")


@dataclass(frozen=True)
class Checkpoint:
    input_position: int
    last_committed_sequence: int | None
    last_committed_sha256: str | None
    completed_group_index: int = -1
    completed_group: tuple[str, str, str] | None = None
    input_identity: dict[str, object] | None = None
    configuration_id: str | None = None


class CheckpointStore:
    def __init__(self, path: Path):
        self.path = path

    def save(self, checkpoint: Checkpoint) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(checkpoint.__dict__, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def load_validated(self, manifest_path: Path) -> Checkpoint:
        checkpoint = Checkpoint(**json.loads(self.path.read_text(encoding="utf-8")))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["shards"]
        if checkpoint.last_committed_sequence is None:
            if checkpoint.last_committed_sha256 is not None:
                raise RuntimeError("checkpoint has hash without committed shard")
            return checkpoint
        matches = [
            item
            for item in manifest
            if item["sequence"] == checkpoint.last_committed_sequence
        ]
        if len(matches) != 1 or matches[0]["sha256"] != checkpoint.last_committed_sha256:
            raise RuntimeError("checkpoint does not match committed shard manifest")
        shard_path = manifest_path.parent / matches[0]["filename"]
        if not shard_path.is_file():
            raise RuntimeError("checkpoint shard is missing")
        actual_sha256 = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        if actual_sha256 != matches[0]["sha256"] or actual_sha256 != checkpoint.last_committed_sha256:
            raise RuntimeError("checkpoint shard checksum mismatch")
        if any(item["sequence"] > checkpoint.last_committed_sequence for item in manifest):
            raise RuntimeError("checkpoint is behind committed output")
        return checkpoint

    def load_or_start(self, manifest_path: Path) -> Checkpoint | None:
        if not self.path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["shards"]
            if manifest:
                raise RuntimeError("output exists without checkpoint")
            return None
        return self.load_validated(manifest_path)


def _read_repaired_stream(
    path: Path,
    sensor: str,
    chunk_size: int,
) -> tuple[pd.DataFrame, list[object]]:
    state = ChunkRepairState()
    emitted: list[pd.DataFrame] = []
    records: list[object] = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        rows, chunk_records = state.process_chunk(chunk, sensor)
        if not rows.empty:
            emitted.append(rows)
        records.extend(chunk_records)
    tail, tail_records = state.finish(sensor)
    if not tail.empty:
        emitted.append(tail)
    records.extend(tail_records)
    if not emitted:
        return pd.read_csv(path, nrows=0), records
    return pd.concat(emitted, ignore_index=True), records


def preprocess_saved_fixture_streaming(
    fixture_directory: Path,
    output_directory: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
    chunk_size: int = 25_000,
) -> dict[str, object]:
    """Run the existing fixture semantics through bounded streaming components."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    streams: dict[str, pd.DataFrame] = {}
    repaired_blocks: list[object] = []
    for sensor, filename in (("acc", "accelerometer.csv"), ("gyro", "gyroscope.csv")):
        streams[sensor], records = _read_repaired_stream(
            fixture_directory / filename, sensor, chunk_size
        )
        repaired_blocks.extend(records)

    groups = set(
        map(
            tuple,
            streams["acc"][list(GROUP_COLUMNS)].drop_duplicates().astype(str).to_numpy(),
        )
    ) & set(
        map(
            tuple,
            streams["gyro"][list(GROUP_COLUMNS)].drop_duplicates().astype(str).to_numpy(),
        )
    )
    writer = CompressedShardWriter(output_directory)
    rejections: list[Rejection] = []
    filter_states = {"acc": FilterState(), "gyro": FilterState()}
    window_index = 0
    for group in sorted(groups):
        group_frames = {
            sensor: streams[sensor][
                (streams[sensor]["User"].astype(str) == group[0])
                & (streams[sensor]["Model"].astype(str) == group[1])
                & (streams[sensor]["Device"].astype(str) == group[2])
            ].reset_index(drop=True)
            for sensor in ("acc", "gyro")
        }
        sessions: dict[str, list[object]] = {}
        for sensor in ("acc", "gyro"):
            sessions[sensor], session_rejections = split_sensor_segments(
                group_frames[sensor], config
            )
            rejections.extend(session_rejections)
        for acc_session in sessions["acc"]:
            for gyro_session in sessions["gyro"]:
                acc_fragments = split_at_interpolation_gaps(acc_session, config)
                gyro_fragments = split_at_interpolation_gaps(gyro_session, config)
                for acc_segment in acc_fragments:
                    filter_states["acc"].begin_segment(str(acc_segment.segment_id))
                    for gyro_segment in gyro_fragments:
                        filter_states["gyro"].begin_segment(str(gyro_segment.segment_id))
                        timestamps, values, labels, alignment_rejections = align_sensor_segments(
                            acc_segment, gyro_segment, config, group
                        )
                        rejections.extend(alignment_rejections)
                        if len(timestamps) == 0:
                            continue
                        windows, window_labels, metadata, window_rejections = generate_windows(
                            timestamps, values, labels, group,
                            f"{acc_segment.segment_id}_{gyro_segment.segment_id}", config
                        )
                        rejections.extend(window_rejections)
                        overlap_buffer = WindowOverlapBuffer(
                            config.samples_per_window, config.step_samples
                        )
                        streamed_windows = overlap_buffer.append(values, labels)
                        if len(streamed_windows) != len(windows):
                            raise RuntimeError("streaming overlap buffer disagrees with baseline window count")
                        for streamed, baseline in zip(streamed_windows, windows):
                            if not np.array_equal(streamed[0], baseline):
                                raise RuntimeError("streaming overlap buffer changed window samples")
                        for window, label, item in zip(windows, window_labels, metadata):
                            trace = {
                                "shard_sequence": 0,
                                "window_index": window_index,
                                "user": item["User"],
                                "model": item["Model"],
                                "device": item["Device"],
                                "source_segment_ids": [str(acc_segment.segment_id), str(gyro_segment.segment_id)],
                                "start_timestamp_raw": item["start_timestamp_raw"],
                                "end_timestamp_raw": item["end_timestamp_raw"],
                                "label": label,
                                "config_version": "har-streaming-fixture-v1",
                            }
                            writer.append(window, label, trace)
                            window_index += 1
    writer.close()
    manifest_path = output_directory / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"shards": []}
    if manifest["shards"]:
        last = manifest["shards"][-1]
        CheckpointStore(output_directory / "checkpoint.json").save(
            Checkpoint(
                input_position=0,
                last_committed_sequence=last["sequence"],
                last_committed_sha256=last["sha256"],
            )
        )
    rejection_counts = pd.Series(
        [item.reason for item in rejections], dtype="string"
    ).value_counts().to_dict()
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "repaired_blocks.json").write_text(
        json.dumps([asdict(record) for record in repaired_blocks], indent=2),
        encoding="utf-8",
    )
    summary = {
        "status": "complete" if window_index else "no_accepted_windows",
        "accepted_windows": window_index,
        "window_shape": [128, 6] if window_index else [0, 0, 6],
        "groups_processed": [list(group) for group in sorted(groups)],
        "rejections": rejection_counts,
        "repaired_block_count": len(repaired_blocks),
        "output_directory": str(output_directory),
    }
    (output_directory / "stream_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def preprocess_stage2_validation_oracle(
    accelerometer_path: Path,
    gyroscope_path: Path,
    output_directory: Path,
    selected_groups: set[tuple[str, str, str]] | None = None,
    config: PreprocessingConfig = PreprocessingConfig(),
    chunk_size: int = 25_000,
    windows_per_shard: int = 1_000,
) -> dict[str, object]:
    """Run the approved Stage 2 validation orchestration on selected groups.

    This is intentionally a validation API: it writes auditable shards and
    diagnostics, but does not normalize data, assign splits, or train models.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if selected_groups is None:
        selected_groups = set()
    output_directory.mkdir(parents=True, exist_ok=True)
    streams: dict[str, pd.DataFrame] = {}
    repaired_blocks: list[object] = []
    for sensor, path in (("acc", accelerometer_path), ("gyro", gyroscope_path)):
        streams[sensor], records = _read_repaired_stream(path, sensor, chunk_size)
        repaired_blocks.extend(records)

    available = {
        sensor: {
            tuple(str(value) for value in row)
            for row in streams[sensor][list(GROUP_COLUMNS)]
            .drop_duplicates()
            .to_numpy()
        }
        for sensor in ("acc", "gyro")
    }
    paired = available["acc"] & available["gyro"]
    groups = sorted(selected_groups & paired)
    writer = CompressedShardWriter(output_directory, windows_per_shard)
    rejections: list[Rejection] = []
    filter_states = {"acc": FilterState(), "gyro": FilterState()}
    accepted = 0
    pair_comparisons = 0
    fragments = 0

    for group in groups:
        group_frames = {
            sensor: streams[sensor][
                (streams[sensor]["User"].astype(str) == group[0])
                & (streams[sensor]["Model"].astype(str) == group[1])
                & (streams[sensor]["Device"].astype(str) == group[2])
            ].reset_index(drop=True)
            for sensor in ("acc", "gyro")
        }
        sessions: dict[str, list[SensorSegment]] = {}
        for sensor in ("acc", "gyro"):
            sessions[sensor], session_rejections = split_sensor_segments(
                group_frames[sensor], config
            )
            rejections.extend(
                Rejection(item.reason, *group, item.sensor_segment)
                for item in session_rejections
            )
        for sensor in ("acc", "gyro"):
            sessions[sensor] = [
                fragment
                for session in sessions[sensor]
                for fragment in split_at_interpolation_gaps(session, config)
            ]
        fragments += len(sessions["acc"]) + len(sessions["gyro"])
        for acc_segment in sessions["acc"]:
            filter_states["acc"].begin_segment(f"acc:{group}:{acc_segment.segment_id}")
            for gyro_segment in sessions["gyro"]:
                filter_states["gyro"].begin_segment(f"gyro:{group}:{gyro_segment.segment_id}")
                pair_comparisons += 1
                timestamps, values, labels, alignment_rejections = align_sensor_segments(
                    acc_segment, gyro_segment, config, group
                )
                rejections.extend(alignment_rejections)
                if len(timestamps) == 0:
                    continue
                segment_id = f"{acc_segment.segment_id}_{gyro_segment.segment_id}"
                windows, window_labels, metadata, window_rejections = generate_windows(
                    timestamps, values, labels, group, segment_id, config
                )
                rejections.extend(window_rejections)
                for window, label, item in zip(windows, window_labels, metadata):
                    source_positions = {
                        "acc": (
                            acc_segment.source_positions.tolist()
                            if acc_segment.source_positions is not None
                            else []
                        ),
                        "gyro": (
                            gyro_segment.source_positions.tolist()
                            if gyro_segment.source_positions is not None
                            else []
                        ),
                    }
                    trace = {
                        **item,
                        "shard_sequence": writer.next_sequence,
                        "window_index": writer.next_window_index,
                        "user": item["User"],
                        "model": item["Model"],
                        "device": item["Device"],
                        "sensor_pair": "acc+gyro",
                        "source_segment_ids": [
                            str(acc_segment.segment_id),
                            str(gyro_segment.segment_id),
                        ],
                        "source_row_positions": source_positions,
                        "config_version": "har-stage2-validation-v1",
                    }
                    writer.append(window, label, trace)
                    accepted += 1
    writer.close()

    manifest_path = output_directory / "shard_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"shards": []}
    )
    checkpoint = None
    if manifest["shards"]:
        last = manifest["shards"][-1]
        checkpoint = Checkpoint(0, last["sequence"], last["sha256"])
        CheckpointStore(output_directory / "checkpoint.json").save(checkpoint)
    rejection_counts = pd.Series(
        [item.reason for item in rejections], dtype="string"
    ).value_counts().to_dict()
    provenance = [asdict(record) for record in repaired_blocks]
    (output_directory / "repaired_blocks.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    summary = {
        "status": "complete",
        "validation_only": True,
        "accepted_windows": accepted,
        "window_shape": [128, 6] if accepted else [0, 0, 6],
        "selected_groups": [list(group) for group in groups],
        "paired_groups_available": [list(group) for group in sorted(paired)],
        "unmatched_accelerometer_groups": [
            list(group) for group in sorted(available["acc"] - available["gyro"])
        ],
        "unmatched_gyroscope_groups": [
            list(group) for group in sorted(available["gyro"] - available["acc"])
        ],
        "fragment_count": fragments,
        "pair_comparisons": pair_comparisons,
        "repaired_block_count": len(repaired_blocks),
        "rejections": rejection_counts,
        "shard_count": len(manifest["shards"]),
        "checkpoint": asdict(checkpoint) if checkpoint else None,
        "output_directory": str(output_directory),
    }
    (output_directory / "stage2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _preprocess_stage2_validation_bounded(
    accelerometer_path: Path,
    gyroscope_path: Path,
    output_directory: Path,
    selected_groups: set[tuple[str, str, str]] | None = None,
    config: PreprocessingConfig = PreprocessingConfig(),
    chunk_size: int = 25_000,
    stage2_config: BoundedStage2Config = BoundedStage2Config(),
) -> dict[str, object]:
    """Run Stage 2 with group-by-group disk spooling and bounded diagnostics."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    if (output_directory / "stage2_summary.json").exists():
        raise RuntimeError("Stage 2 output already exists; resume is fail-closed")
    checkpoint_path = output_directory / "checkpoint.json"
    manifest_path = output_directory / "shard_manifest.json"
    existing_checkpoint = checkpoint_path.exists()
    if existing_checkpoint and not manifest_path.exists():
        raise RuntimeError("checkpoint exists without shard manifest")
    if not existing_checkpoint and manifest_path.exists():
        raise RuntimeError("output exists without checkpoint")
    input_identity = {
        "accelerometer": _input_identity(accelerometer_path),
        "gyroscope": _input_identity(gyroscope_path),
    }
    rejections_path = output_directory / "rejections.jsonl.gz"
    eligibility_path = output_directory / "group_eligibility.jsonl.gz"
    pair_path = output_directory / "pair_diagnostics.jsonl.gz"
    repair_path = output_directory / "repair_provenance.jsonl.gz"
    with tempfile.TemporaryDirectory(prefix="dw-dl-stage2-") as temporary:
        spool_root = Path(temporary)
        spool_dirs = {"acc": spool_root / "acc", "gyro": spool_root / "gyro"}
        spool_dirs["acc"].mkdir()
        spool_dirs["gyro"].mkdir()
        with (
            gzip.open(repair_path, "at" if existing_checkpoint else "wt", encoding="utf-8") as repair_handle,
            gzip.open(rejections_path, "at" if existing_checkpoint else "wt", encoding="utf-8") as rejection_handle,
            gzip.open(pair_path, "at" if existing_checkpoint else "wt", encoding="utf-8") as pair_handle,
        ):
            acc_groups = _spool_repaired_stream(
                accelerometer_path, "acc", spool_dirs["acc"],
                repair_handle if not existing_checkpoint else None,
                chunk_size, stage2_config,
                selected_groups,
            )
            gyro_groups = _spool_repaired_stream(
                gyroscope_path, "gyro", spool_dirs["gyro"],
                repair_handle if not existing_checkpoint else None,
                chunk_size, stage2_config,
                selected_groups,
            )
            paired_groups = acc_groups & gyro_groups
            groups = sorted(
                paired_groups if selected_groups is None
                else paired_groups & selected_groups
            )
            configuration_id = _stage2_configuration_id(
                config, stage2_config, chunk_size, groups
            )
            writer = CompressedShardWriter(
                output_directory, stage2_config.windows_per_shard
            )
            if existing_checkpoint:
                checkpoint = CheckpointStore(checkpoint_path).load_validated(manifest_path)
                if checkpoint.input_identity != input_identity:
                    raise RuntimeError("checkpoint input identity mismatch")
                if checkpoint.configuration_id != configuration_id:
                    raise RuntimeError("checkpoint configuration mismatch")
                if checkpoint.completed_group_index >= len(groups):
                    raise RuntimeError("checkpoint group position is ahead of output")
                if checkpoint.completed_group is not None and (
                    tuple(groups[checkpoint.completed_group_index])
                    != tuple(checkpoint.completed_group)
                ):
                    raise RuntimeError("checkpoint group identity mismatch")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                writer._manifest = manifest["shards"]
                writer._sequence = (
                    manifest["shards"][-1]["sequence"] + 1
                    if manifest["shards"] else 0
                )
                accepted = sum(item["window_count"] for item in manifest["shards"])
                start_group_index = checkpoint.completed_group_index + 1
            else:
                start_group_index = 0
                writer._write_manifest()
            accepted = 0
            if existing_checkpoint:
                accepted = sum(item["window_count"] for item in manifest["shards"])
            fragments = 0
            pair_comparisons = 0
            rejection_counts: dict[str, int] = {}
            eligibility_count = 0
            group_paths = {
                sensor: {
                    group: next(
                        path for path in spool_dirs[sensor].glob("*.csv")
                        if path.stem.startswith(f"{sensor}-")
                        and tuple(
                            str(value)
                            for value in pd.read_csv(path, nrows=1)[list(GROUP_COLUMNS)]
                            .iloc[0]
                        ) == group
                    )
                    for group in (acc_groups if sensor == "acc" else gyro_groups)
                }
                for sensor in ("acc", "gyro")
            }

            def reject(item: Rejection) -> None:
                rejection_counts[item.reason] = rejection_counts.get(item.reason, 0) + 1
                rejection_handle.write(json.dumps(asdict(item), sort_keys=True) + "\n")

            for group_index, group in enumerate(groups):
                if group_index < start_group_index:
                    continue
                _check_stage2_rss(stage2_config)
                group_fragments: dict[str, list[SensorSegment]] = {}
                for sensor in ("acc", "gyro"):
                    _check_stage2_rss(stage2_config)
                    frame = pd.read_csv(group_paths[sensor][group])
                    sessions, errors = split_sensor_segments(frame, config)
                    for error in errors:
                        reject(Rejection(error.reason, *group, error.sensor_segment))
                    group_fragments[sensor] = [
                        fragment
                        for session in sessions
                        for fragment in split_at_interpolation_gaps(session, config)
                    ]
                    del frame, sessions
                fragments += sum(len(items) for items in group_fragments.values())
                candidate_pairs = interval_sweep_overlaps(
                    group_fragments["acc"], group_fragments["gyro"]
                )
                group_accepted = 0
                for acc_segment, gyro_segment in candidate_pairs:
                    pair_comparisons += 1
                    if (
                        len(acc_segment.timestamp) > stage2_config.max_disk_backed_fragment_samples
                        or len(gyro_segment.timestamp) > stage2_config.max_disk_backed_fragment_samples
                    ):
                        reject(Rejection(
                            "fragment_exceeds_supported_filter_length", *group,
                            f"{acc_segment.segment_id}_{gyro_segment.segment_id}",
                        ))
                        continue
                    if (
                        len(acc_segment.timestamp) > stage2_config.max_in_memory_fragment_samples
                        or len(gyro_segment.timestamp) > stage2_config.max_in_memory_fragment_samples
                    ):
                        reject(Rejection(
                            "fragment_exceeds_supported_filter_length", *group,
                            f"{acc_segment.segment_id}_{gyro_segment.segment_id}",
                        ))
                        continue
                    _check_stage2_rss(stage2_config)
                    timestamps, values, labels, errors = align_sensor_segments(
                        acc_segment, gyro_segment, config, group
                    )
                    for error in errors:
                        reject(error)
                    pair_path_record = {
                        "user": group[0], "model": group[1], "device": group[2],
                        "acc_segment": str(acc_segment.segment_id),
                        "gyro_segment": str(gyro_segment.segment_id),
                        "overlap_samples": int(len(timestamps)),
                    }
                    pair_handle.write(json.dumps(pair_path_record, sort_keys=True) + "\n")
                    if not len(timestamps):
                        continue
                    windows, window_labels, metadata, errors = generate_windows(
                        timestamps, values, labels, group,
                        f"{acc_segment.segment_id}_{gyro_segment.segment_id}", config,
                    )
                    for error in errors:
                        reject(error)
                    for window, label, item in zip(windows, window_labels, metadata):
                        trace = {
                            **item,
                            "user": item["User"], "model": item["Model"],
                            "device": item["Device"],
                            "shard_sequence": writer.next_sequence,
                            "window_index": writer.next_window_index,
                            "sensor_pair": "acc+gyro",
                            "source_segment_ids": [
                                str(acc_segment.segment_id), str(gyro_segment.segment_id)
                            ],
                            "source_traceability_ref": (
                                f"{group[0]}|{group[1]}|{group[2]}|"
                                f"acc:{acc_segment.segment_id}|gyro:{gyro_segment.segment_id}"
                            ),
                            "input_shard": {
                                "acc": accelerometer_path.name,
                                "gyro": gyroscope_path.name,
                            },
                            "source_sample_counts": {
                                "acc": int(len(acc_segment.timestamp)),
                                "gyro": int(len(gyro_segment.timestamp)),
                            },
                            "target_sample_count": int(len(window)),
                            "aligned_target_grid_sample_count": int(len(timestamps)),
                            "resampling": {
                                "target_hz": config.target_hz,
                                "target_step_raw": int(
                                    round(1_000_000_000 / config.target_hz)
                                ),
                                "source_rate_hz": {
                                    "acc": _median_rate_hz(acc_segment.timestamp),
                                    "gyro": _median_rate_hz(gyro_segment.timestamp),
                                },
                                "filter_semantics": "complete-fragment-zero-phase",
                            },
                            "config_version": "har-stage2-bounded-v1",
                        }
                        writer.append(window, label, trace)
                        accepted += 1
                        group_accepted += 1
                eligibility_record = {
                    "user": group[0], "model": group[1], "device": group[2],
                    "post_repair_fragment_count": sum(
                        len(items) for items in group_fragments.values()
                    ),
                    "overlapping_pair_count": len(candidate_pairs),
                    "accepted_window_count": group_accepted,
                    "pre_repair_pilot_selected": selected_groups is not None
                    and group in selected_groups,
                }
                eligibility_count += 1
                with gzip.open(eligibility_path, "at", encoding="utf-8") as handle:
                    handle.write(json.dumps(eligibility_record, sort_keys=True) + "\n")
                writer.flush()
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                last = manifest["shards"][-1] if manifest["shards"] else None
                CheckpointStore(checkpoint_path).save(
                    Checkpoint(
                        input_position=group_index,
                        last_committed_sequence=last["sequence"] if last else None,
                        last_committed_sha256=last["sha256"] if last else None,
                        completed_group_index=group_index,
                        completed_group=group,
                        input_identity=input_identity,
                        configuration_id=configuration_id,
                    )
                )
                del group_fragments, candidate_pairs
                _check_stage2_rss(stage2_config)
            writer.close()

    manifest_path = output_directory / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"shards": []}
    checkpoint = None
    if manifest["shards"]:
        last = manifest["shards"][-1]
        checkpoint = Checkpoint(0, last["sequence"], last["sha256"])
        CheckpointStore(output_directory / "checkpoint.json").save(checkpoint)
    summary = {
        "status": "complete",
        "validation_only": True,
        "accepted_windows": accepted,
        "window_shape": [128, 6] if accepted else [0, 0, 6],
        "paired_groups_processed": [list(group) for group in groups],
        "paired_groups_available_after_repair": [list(group) for group in sorted(paired_groups)],
        "unmatched_accelerometer_groups": [list(group) for group in sorted(acc_groups - gyro_groups)],
        "unmatched_gyroscope_groups": [list(group) for group in sorted(gyro_groups - acc_groups)],
        "fragment_count": fragments,
        "pair_comparisons": pair_comparisons,
        "rejections": rejection_counts,
        "group_eligibility_count": eligibility_count,
        "shard_count": len(manifest["shards"]),
        "checkpoint": asdict(checkpoint) if checkpoint else None,
        "configuration": asdict(stage2_config),
        "output_directory": str(output_directory),
    }
    (output_directory / "stage2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def resample_and_window_stage2(
    repaired_accelerometer_path: Path,
    repaired_gyroscope_path: Path,
    output_directory: Path,
    selected_groups: set[tuple[str, str, str]] | None = None,
    config: PreprocessingConfig = PreprocessingConfig(),
    chunk_size: int = 25_000,
    stage2_config: BoundedStage2Config = BoundedStage2Config(),
) -> dict[str, object]:
    """Create validated raw windows from repaired Stage 2 sensor streams."""
    return preprocess_stage2_validation(
        repaired_accelerometer_path,
        repaired_gyroscope_path,
        output_directory,
        selected_groups,
        config,
        chunk_size,
        stage2_config,
    )


def preprocess_stage2_validation(
    accelerometer_path: Path,
    gyroscope_path: Path,
    output_directory: Path,
    selected_groups: set[tuple[str, str, str]] | None = None,
    config: PreprocessingConfig = PreprocessingConfig(),
    chunk_size: int = 25_000,
    stage2_config: BoundedStage2Config = BoundedStage2Config(),
) -> dict[str, object]:
    try:
        return _preprocess_stage2_validation_bounded(
            accelerometer_path,
            gyroscope_path,
            output_directory,
            selected_groups,
            config,
            chunk_size,
            stage2_config,
        )
    except MemoryError as error:
        output_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = output_directory / "shard_manifest.json"
        checkpoint = None
        checkpoint_path = output_directory / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        (output_directory / "stage2_failure.json").write_text(
            json.dumps(
                {
                    "status": "failed_closed",
                    "reason": "rss_limit_exceeded",
                    "error": str(error),
                    "checkpoint": asdict(checkpoint) if checkpoint else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
