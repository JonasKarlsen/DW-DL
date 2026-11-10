"""Bounded streaming primitives for fixture-sized preprocessing experiments."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .har_preprocessing import GROUP_COLUMNS, repair_adjacent_swaps

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
            previous_boundary = self.boundary_context.get(group_key)
            if previous_boundary is not None and previous_boundary != boundary:
                old_key = (*group_key, previous_boundary)
                emitted.extend(self.pending_rows.pop(old_key, []))
                self.pending_eligible_pair.pop(old_key, None)
            self.boundary_context[group_key] = boundary
            key = (*group_key, boundary)
            buffer = self.pending_rows.setdefault(key, [])
            buffer.append(row.copy())
            self.group_context[group_key] = (sensor, boundary)
            if len(buffer) < 3:
                self._update_pending_flag(key)
                continue
            candidate = pd.DataFrame(buffer)
            repaired, candidate_records = repair_adjacent_swaps(candidate, sensor)
            emitted.append(repaired.iloc[0].copy())
            records.extend(candidate_records)
            self.pending_rows[key] = [
                repaired.iloc[1].copy(),
                repaired.iloc[2].copy(),
            ]
            self._update_pending_flag(key)
        if not emitted:
            return chunk.iloc[0:0].copy(), records
        return pd.DataFrame(emitted, columns=chunk.columns), records

    def finish(self, sensor: str) -> tuple[pd.DataFrame, list[object]]:
        emitted: list[pd.Series] = []
        records: list[object] = []
        for key in list(self.pending_rows):
            buffer = self.pending_rows.pop(key)
            if buffer:
                repaired, candidate_records = repair_adjacent_swaps(
                    pd.DataFrame(buffer), sensor
                )
                emitted.extend(repaired.iloc[index].copy() for index in range(len(repaired)))
                records.extend(candidate_records)
            self.pending_eligible_pair.pop(key, None)
            self.group_context.pop(key[:3], None)
            self.boundary_context.pop(key[:3], None)
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
