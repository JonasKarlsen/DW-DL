import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dw_dl.har_preprocessing import (
    PreprocessingConfig,
    SensorSegment,
    _resample_values,
    align_sensor_segments,
    generate_windows,
    make_subject_splits,
    split_sensor_segments,
    split_at_interpolation_gaps,
    repair_adjacent_swaps,
    preprocess_saved_fixture,
)
from dw_dl.har_streaming import (
    Checkpoint,
    CheckpointStore,
    BoundedStage2Config,
    STAGE2_RSS_LIMIT_BYTES,
    ChunkRepairState,
    CompressedShardWriter,
    FilterState,
    WindowOverlapBuffer,
    _read_repaired_stream,
    _spool_repaired_stream,
    interval_sweep_overlaps,
    preprocess_stage2_validation,
    resample_and_window_stage2,
    preprocess_saved_fixture_streaming,
)
import dw_dl.har_streaming as har_streaming


def _segment(rate: float, offset: float = 0.0, duration: float = 8.0) -> SensorSegment:
    seconds = np.arange(0, duration, 1 / rate) + offset
    timestamps = np.rint(seconds * 1_000_000_000).astype(np.int64)
    values = np.column_stack([np.sin(seconds * 2 * np.pi * 2 + i) for i in range(3)])
    labels = np.full(len(seconds), "walk", dtype=object)
    return SensorSegment(timestamps, values, labels, 0, int(timestamps[0]), int(timestamps[-1]))


def test_unequal_rates_and_offset_share_grid():
    config = PreprocessingConfig(minimum_segment_overlap_seconds=2.56)
    timestamps, values, labels, rejections = align_sensor_segments(
        _segment(100), _segment(200, 0.01), config
    )
    assert not rejections
    assert values.shape[1] == 6
    assert np.all(np.diff(timestamps) > 0)
    assert np.median(np.diff(timestamps)) == 20_000_000
    assert np.all(labels == "walk")


def test_antialias_filter_attenuates_above_target_nyquist():
    source_rate = 100.0
    seconds = np.arange(0, 5, 1 / source_rate)
    source_timestamps = np.rint(seconds * 1_000_000_000).astype(np.int64)
    source = np.sin(2 * np.pi * 40 * seconds)[:, None]
    target_timestamps = np.arange(0, 5, 1 / 50) * 1_000_000_000
    target_timestamps = target_timestamps.astype(np.int64)
    filtered = _resample_values(
        source_timestamps,
        source,
        target_timestamps,
        50,
        0.4,
    )
    assert float(np.std(filtered[10:-10])) < 0.2


def test_gaps_and_resets_make_separate_segments():
    base = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64) * 1_000_000_000
    timestamps = np.r_[base[:3], base[3:] + 10_000_000_000]
    frame = pd.DataFrame(
        {
            "Creation_Time": timestamps,
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "gt": "walk",
        }
    )
    segments, _ = split_sensor_segments(
        frame, PreprocessingConfig(session_gap_seconds=5)
    )
    assert len(segments) == 2


def test_mixed_labels_are_rejected():
    config = PreprocessingConfig()
    timestamps = np.arange(128, dtype=np.int64) * 20_000_000
    values = np.zeros((128, 6), dtype=np.float32)
    labels = np.array(["walk"] * 64 + ["sit"] * 64, dtype=object)
    windows, _, _, rejections = generate_windows(
        timestamps, values, labels, ("a", "m", "d"), "0", config
    )
    assert not windows
    assert any(item.reason == "mixed_label" for item in rejections)


def test_missing_sensor_category_counts():
    quality = pd.DataFrame(
        {
            "User": ["a", "a", "b"],
            "Model": ["m", "m", "m"],
            "Device": ["d", "d", "e"],
            "Sensor": ["acc", "gyro", "acc"],
        }
    )
    acc = set(map(tuple, quality[quality.Sensor == "acc"][["User", "Model", "Device"]].to_numpy()))
    gyro = set(map(tuple, quality[quality.Sensor == "gyro"][["User", "Model", "Device"]].to_numpy()))
    assert len(acc - gyro) == 1
    assert len(gyro - acc) == 0


def test_global_subject_splits_are_disjoint():
    splits = make_subject_splits(["a", "b", "c", "d", "e"], seed=1)
    assert len(splits) == 5
    assert set(splits) == {"a", "b", "c", "d", "e"}
    assert set(splits.values()) == {"train", "val", "test"}


def test_duplicate_timestamp_is_not_silently_repaired():
    frame = pd.DataFrame(
        {
            "Creation_Time": [0, 20_000_000, 20_000_000, 40_000_000],
            "x": [1.0] * 4,
            "y": [1.0] * 4,
            "z": [1.0] * 4,
            "gt": ["walk"] * 4,
        }
    )
    segments, rejections = split_sensor_segments(frame, PreprocessingConfig())
    assert sum(item.reason == "clock_reset_or_out_of_order" for item in rejections) == 1
    assert sum(len(segment.timestamp) for segment in segments) == 4


def test_nonfinite_sensor_value_is_rejected_as_a_boundary():
    frame = pd.DataFrame(
        {
            "Creation_Time": [0, 20_000_000, 40_000_000, 60_000_000],
            "x": [1.0, np.nan, 1.0, 1.0],
            "y": [1.0] * 4,
            "z": [1.0] * 4,
            "gt": ["walk"] * 4,
        }
    )
    segments, rejections = split_sensor_segments(frame, PreprocessingConfig())
    assert any(item.reason == "nonfinite_sensor_value" for item in rejections)
    assert sum(len(segment.timestamp) for segment in segments) == 2


def test_small_out_of_order_block_is_reported_as_boundary():
    frame = pd.DataFrame(
        {
            "Creation_Time": [0, 20_000_000, 15_000_000, 40_000_000],
            "x": [1.0] * 4,
            "y": [1.0] * 4,
            "z": [1.0] * 4,
            "gt": ["walk"] * 4,
        }
    )
    segments, rejections = split_sensor_segments(frame, PreprocessingConfig())
    assert any(item.reason == "clock_reset_or_out_of_order" for item in rejections)
    assert len(segments) == 2


def test_large_backward_jump_is_reported_without_repair():
    frame = pd.DataFrame(
        {
            "Creation_Time": [
                0,
                20_000_000,
                10_000_000_000,
                1_000_000_000,
                1_020_000_000,
            ],
            "x": [1.0] * 5,
            "y": [1.0] * 5,
            "z": [1.0] * 5,
            "gt": ["walk"] * 5,
        }
    )
    segments, rejections = split_sensor_segments(frame, PreprocessingConfig())
    assert any(item.reason == "clock_reset_or_out_of_order" for item in rejections)
    assert sum(len(segment.timestamp) for segment in segments) == 4


def test_fragmented_overlap_has_no_eligible_pair():
    config = PreprocessingConfig(minimum_segment_overlap_seconds=2.56)
    acc = _segment(100, duration=2.0)
    gyro = _segment(200, duration=2.0)
    _, _, _, rejections = align_sensor_segments(acc, gyro, config)
    assert [item.reason for item in rejections] == ["insufficient_intersection"]


def _swap_frame(times, indices, arrivals=None):
    return pd.DataFrame(
        {
            "Index": indices,
            "Arrival_Time": arrivals or [100] * len(times),
            "Creation_Time": times,
            "x": np.arange(len(times), dtype=float),
            "y": np.arange(len(times), dtype=float) + 10,
            "z": np.arange(len(times), dtype=float) + 20,
            "gt": ["walk"] * len(times),
        }
    )


def test_verified_adjacent_swap_is_repaired_with_provenance():
    frame = _swap_frame([0, 30, 20, 40], [0, 2, 1, 3])
    repaired, records = repair_adjacent_swaps(frame, "acc")
    assert repaired.Creation_Time.tolist() == [0, 20, 30, 40]
    assert repaired.Index.tolist() == [0, 1, 2, 3]
    assert repaired.x.tolist() == [0, 2, 1, 3]
    assert len(records) == 1
    assert records[0].original_positions == (1, 2)


def test_ordered_sequence_is_unchanged_and_repair_is_idempotent():
    frame = _swap_frame([0, 20, 30, 40], [0, 1, 2, 3])
    repaired, records = repair_adjacent_swaps(frame, "acc")
    again, second_records = repair_adjacent_swaps(repaired, "acc")
    pd.testing.assert_frame_equal(frame, repaired)
    pd.testing.assert_frame_equal(repaired, again)
    assert records == []
    assert second_records == []


def test_ambiguous_reversal_is_not_repaired():
    frame = _swap_frame([0, 30, 20, 40], [0, 2, 1, 3], [100, 101, 102, 103])
    repaired, records = repair_adjacent_swaps(frame, "acc")
    pd.testing.assert_frame_equal(frame, repaired)
    assert records == []


def test_large_reset_is_not_repaired_and_rows_are_preserved():
    frame = _swap_frame([0, 10_000_000_000, 1_000_000_000, 11_000_000_000], [0, 1, 2, 3])
    repaired, records = repair_adjacent_swaps(frame, "gyro")
    pd.testing.assert_frame_equal(frame, repaired)
    assert records == []
    assert len(repaired) == len(frame)
    assert repaired.Creation_Time.tolist() == frame.Creation_Time.tolist()


def test_saved_repair_provenance_contains_original_row_payload(tmp_path):
    rows = [
        {
            "Index": 0,
            "Arrival_Time": 100,
            "Creation_Time": 0,
            "x": 10.0,
            "y": 11.0,
            "z": 12.0,
            "User": "a",
            "Model": "m",
            "Device": "d",
            "gt": "walk",
        },
        {
            "Index": 2,
            "Arrival_Time": 101,
            "Creation_Time": 30_000_000,
            "x": 20.0,
            "y": 21.0,
            "z": 22.0,
            "User": "a",
            "Model": "m",
            "Device": "d",
            "gt": "walk",
        },
        {
            "Index": 1,
            "Arrival_Time": 101,
            "Creation_Time": 20_000_000,
            "x": 30.0,
            "y": 31.0,
            "z": 32.0,
            "User": "a",
            "Model": "m",
            "Device": "d",
            "gt": "walk",
        },
    ]
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    pd.DataFrame(rows).to_csv(fixture / "accelerometer.csv", index=False)
    pd.DataFrame(rows).to_csv(fixture / "gyroscope.csv", index=False)

    output = tmp_path / "output"
    preprocess_saved_fixture(fixture, output)
    records = json.loads((output / "repaired_blocks.json").read_text())

    acc_record = next(record for record in records if record["sensor"] == "acc")
    assert acc_record["original_rows"] == [rows[1], rows[2]]
    assert acc_record["original_rows"][0]["x"] == 20.0
    assert acc_record["original_rows"][1]["x"] == 30.0


def test_windows_do_not_cross_interpolation_gap_boundary():
    config = PreprocessingConfig(
        minimum_segment_overlap_seconds=2.56,
        interpolation_gap_seconds=0.25,
    )
    block = np.arange(129, dtype=np.int64) * 20_000_000
    second_block = block + block[-1] + 1_000_000_000
    timestamps = np.concatenate((block, second_block))
    frame = pd.DataFrame(
        {
            "Creation_Time": timestamps,
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "gt": "walk",
        }
    )
    sessions, _ = split_sensor_segments(frame, config)
    fragments = split_at_interpolation_gaps(sessions[0], config)
    assert len(fragments) == 2
    assert (fragments[1].timestamp[0] - fragments[0].timestamp[-1]) / 1e9 > 0.25

    windows = []
    metadata = []
    for fragment in fragments:
        result = generate_windows(
            fragment.timestamp,
            np.column_stack((fragment.values, fragment.values)),
            fragment.labels,
            ("a", "m", "d"),
            str(fragment.segment_id),
            config,
        )
        windows.extend(result[0])
        metadata.extend(result[2])

    assert len(windows) == 2
    assert all(window.shape == (128, 6) for window in windows)
    assert [item["segment_id"] for item in metadata] == ["0.0", "0.1"]


def _stream_rows(times, indices):
    return pd.DataFrame(
        {
            "Index": indices,
            "Arrival_Time": [100] * len(times),
            "Creation_Time": times,
            "x": np.arange(len(times), dtype=float),
            "y": np.arange(len(times), dtype=float) + 10,
            "z": np.arange(len(times), dtype=float) + 20,
            "User": ["a"] * len(times),
            "Model": ["m"] * len(times),
            "Device": ["d"] * len(times),
            "gt": ["walk"] * len(times),
        }
    )


def test_chunk_repair_spanning_boundary_preserves_rows():
    frame = _stream_rows([0, 30, 20, 40], [0, 2, 1, 3])
    state = ChunkRepairState()
    first, records = state.process_chunk(frame.iloc[:2], "acc")
    second, more_records = state.process_chunk(frame.iloc[2:], "acc")
    tail, final_records = state.finish("acc")
    result = pd.concat([first, second, tail], ignore_index=True)
    assert result.Creation_Time.tolist() == [0, 20, 30, 40]
    assert result.x.tolist() == [0, 2, 1, 3]
    assert len(records + more_records + final_records) == 1


def test_chunk_repair_has_no_loss_duplication_or_unintended_reordering():
    frame = _stream_rows([0, 30, 20, 40, 60], [0, 2, 1, 3, 4])
    state = ChunkRepairState()
    emitted = []
    for chunk in (frame.iloc[:1], frame.iloc[1:3], frame.iloc[3:]):
        output, _ = state.process_chunk(chunk, "acc")
        emitted.append(output)
    emitted.append(state.finish("acc")[0])
    result = pd.concat(emitted, ignore_index=True)
    assert len(result) == len(frame)
    assert sorted(result.Index.tolist()) == sorted(frame.Index.tolist())
    assert result.Creation_Time.tolist() == [0, 20, 30, 40, 60]


def test_filter_state_resets_at_segment_boundaries():
    state = FilterState()
    state.begin_segment("segment-1")
    state.begin_segment("segment-1")
    state.begin_segment("segment-2")
    assert state.reset_count == 2
    assert state.segment_id == "segment-2"


def test_overlapping_windows_continue_across_chunks():
    buffer = WindowOverlapBuffer(4, 2)
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    first = buffer.append(values[:3], np.array(["walk"] * 3))
    second = buffer.append(values[3:], np.array(["walk"]))
    assert len(first) == 0
    assert len(second) == 1
    assert second[0][0][:, 0].tolist() == [0, 6, 12, 18]
    assert buffer.values.shape[0] == 2


def test_interrupted_resume_has_no_duplicate_shard_or_window(tmp_path):
    writer = CompressedShardWriter(tmp_path, windows_per_shard=2)
    for value in (1.0, 2.0):
        writer.append(
            np.full((2, 6), value),
            "walk",
            {
                "shard_sequence": 0,
                "window_index": int(value - 1),
                "user": "a",
                "model": "m",
                "device": "d",
                "source_segment_ids": ["acc-0", "gyro-0"],
                "start_timestamp_raw": 0,
                "end_timestamp_raw": 1,
                "label": "walk",
                "config_version": "test-v1",
            },
        )
    writer.close()
    manifest = json.loads((tmp_path / "shard_manifest.json").read_text())
    shard = manifest["shards"][0]
    checkpoint = Checkpoint(2, shard["sequence"], shard["sha256"])
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save(checkpoint)
    assert store.load_validated(tmp_path / "shard_manifest.json") == checkpoint
    assert len(manifest["shards"]) == 1
    assert np.load(tmp_path / shard["filename"])["X"].shape[0] == 2


def test_corrupted_checkpoint_fails_closed(tmp_path):
    writer = CompressedShardWriter(tmp_path, windows_per_shard=1)
    writer.append(
        np.zeros((2, 6)),
        "walk",
        {
            "shard_sequence": 0,
            "window_index": 0,
            "user": "a",
            "model": "m",
            "device": "d",
            "sensor_pair": "acc+gyro",
            "start_timestamp_raw": 0,
            "end_timestamp_raw": 1,
            "label": "walk",
            "config_version": "test-v1",
        },
    )
    writer.close()
    manifest_path = tmp_path / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save(Checkpoint(1, 0, "corrupted"))
    try:
        store.load_validated(manifest_path)
    except RuntimeError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("corrupted checkpoint was accepted")


def test_modified_shard_fails_fresh_checksum_validation(tmp_path):
    writer = CompressedShardWriter(tmp_path, windows_per_shard=1)
    writer.append(
        np.zeros((2, 6)),
        "walk",
        {
            "shard_sequence": 0, "window_index": 0, "user": "a", "model": "m",
            "device": "d", "sensor_pair": "acc+gyro", "start_timestamp_raw": 0,
            "end_timestamp_raw": 1, "label": "walk", "config_version": "test-v1",
        },
    )
    writer.close()
    manifest = json.loads((tmp_path / "shard_manifest.json").read_text())
    shard_path = tmp_path / manifest["shards"][0]["filename"]
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save(Checkpoint(1, 0, manifest["shards"][0]["sha256"]))
    try:
        store.load_validated(tmp_path / "shard_manifest.json")
    except RuntimeError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("modified shard was accepted")


def test_checkpoint_behind_output_fails_closed(tmp_path):
    writer = CompressedShardWriter(tmp_path, windows_per_shard=1)
    metadata = {
        "shard_sequence": 0, "window_index": 0, "user": "a", "model": "m",
        "device": "d", "sensor_pair": "acc+gyro", "start_timestamp_raw": 0,
        "end_timestamp_raw": 1, "label": "walk", "config_version": "test-v1",
    }
    writer.append(np.zeros((2, 6)), "walk", metadata)
    writer.flush()
    metadata["shard_sequence"] = 1
    writer.append(np.zeros((2, 6)), "walk", metadata)
    writer.close()
    manifest = json.loads((tmp_path / "shard_manifest.json").read_text())
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save(Checkpoint(1, 0, manifest["shards"][0]["sha256"]))
    try:
        store.load_validated(tmp_path / "shard_manifest.json")
    except RuntimeError as error:
        assert "behind" in str(error)
    else:
        raise AssertionError("checkpoint behind output was accepted")


def test_checkpoint_ahead_of_output_fails_closed(tmp_path):
    writer = CompressedShardWriter(tmp_path, windows_per_shard=1)
    writer.append(
        np.zeros((2, 6)),
        "walk",
        {
            "shard_sequence": 0, "window_index": 0, "user": "a", "model": "m",
            "device": "d", "sensor_pair": "acc+gyro", "start_timestamp_raw": 0,
            "end_timestamp_raw": 1, "label": "walk", "config_version": "test-v1",
        },
    )
    writer.close()
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.save(Checkpoint(1, 1, "missing-shard"))
    try:
        store.load_validated(tmp_path / "shard_manifest.json")
    except RuntimeError as error:
        assert "manifest" in str(error)
    else:
        raise AssertionError("checkpoint ahead of output was accepted")


def test_no_checkpoint_starts_only_with_empty_output(tmp_path):
    manifest_path = tmp_path / "shard_manifest.json"
    manifest_path.write_text(json.dumps({"shards": []}))
    assert CheckpointStore(tmp_path / "checkpoint.json").load_or_start(manifest_path) is None


def test_incomplete_metadata_is_rejected(tmp_path):
    writer = CompressedShardWriter(tmp_path)
    try:
        writer.append(np.zeros((2, 6)), "walk", {"user": "a"})
    except ValueError as error:
        assert "metadata" in str(error)
    else:
        raise AssertionError("incomplete metadata was accepted")


FIXTURE_DIRECTORY = Path("code/har_smoke_fixture_v2")
V2_OUTPUT = Path("code/har_smoke_preprocessed_repaired_v2")


def test_streaming_fixture_repair_matches_existing_path():
    original = pd.read_csv(FIXTURE_DIRECTORY / "accelerometer.csv")
    streamed, records = _read_repaired_stream(
        FIXTURE_DIRECTORY / "accelerometer.csv", "acc", chunk_size=101
    )
    expected, expected_records = repair_adjacent_swaps(original, "acc")
    pd.testing.assert_frame_equal(streamed, expected)
    assert [record.original_rows for record in records] == [
        record.original_rows for record in expected_records
    ]


def test_streaming_fixture_filter_resets_preserve_segment_boundaries(tmp_path):
    summary = preprocess_saved_fixture_streaming(
        FIXTURE_DIRECTORY, tmp_path / "streamed", chunk_size=101
    )
    assert summary["accepted_windows"] == 2
    assert summary["rejections"] == json.loads(
        (V2_OUTPUT / "rejections.json").read_text()
    )


def test_streaming_fixture_overlap_matches_two_v2_windows(tmp_path):
    output = tmp_path / "streamed"
    preprocess_saved_fixture_streaming(FIXTURE_DIRECTORY, output, chunk_size=101)
    with np.load(output / "windows-000000.npz") as shard:
        streamed = shard["X"]
    expected = np.load(V2_OUTPUT / "X_raw.npy")
    assert streamed.shape == (2, 128, 6)
    np.testing.assert_array_equal(streamed, expected)


def test_streaming_fixture_shard_contains_exactly_two_windows(tmp_path):
    output = tmp_path / "streamed"
    preprocess_saved_fixture_streaming(FIXTURE_DIRECTORY, output, chunk_size=101)
    manifest = json.loads((output / "shard_manifest.json").read_text())
    assert len(manifest["shards"]) == 1
    assert manifest["shards"][0]["window_count"] == 2


def test_streaming_fixture_checkpoint_resume_has_no_duplicates(tmp_path):
    output = tmp_path / "streamed"
    preprocess_saved_fixture_streaming(FIXTURE_DIRECTORY, output, chunk_size=101)
    store = CheckpointStore(output / "checkpoint.json")
    checkpoint = store.load_validated(output / "shard_manifest.json")
    assert checkpoint.last_committed_sequence == 0
    with np.load(output / "windows-000000.npz") as shard:
        assert shard["X"].shape[0] == 2


def test_streaming_fixture_metadata_matches_v2_semantics(tmp_path):
    output = tmp_path / "streamed"
    preprocess_saved_fixture_streaming(FIXTURE_DIRECTORY, output, chunk_size=101)
    with np.load(output / "windows-000000.npz") as shard:
        records = [json.loads(item) for item in shard["metadata"]]
        assert shard["labels"].tolist() == ["stand", "stand"]
    expected = pd.read_csv(V2_OUTPUT / "metadata.csv")
    assert [record["source_segment_ids"] for record in records] == [
        ["2.0", "1.0"], ["2.1", "1.1"]
    ]
    assert [record["start_timestamp_raw"] for record in records] == (
        expected["start_timestamp_raw"].tolist()
    )


def _write_unequal_rate_fixture(directory: Path) -> tuple[Path, float, float]:
    directory.mkdir()
    rows_by_sensor = {}
    rates = {"accelerometer.csv": 100.0, "gyroscope.csv": 80.0}
    for filename, rate in rates.items():
        seconds = np.arange(0, 6, 1 / rate)
        seconds = seconds[(seconds < 3.0) | (seconds >= 3.5)]
        timestamps = np.rint(seconds * 1_000_000_000).astype(np.int64)
        rows_by_sensor[filename] = pd.DataFrame(
            {
                "Index": np.arange(len(seconds)),
                "Arrival_Time": timestamps // 1_000_000,
                "Creation_Time": timestamps,
                "x": np.sin(seconds),
                "y": np.cos(seconds),
                "z": seconds,
                "User": "synthetic",
                "Model": "unequal",
                "Device": "paired-1",
                "gt": "walk",
            }
        )
        rows_by_sensor[filename].to_csv(directory / filename, index=False)
    return directory, rates["accelerometer.csv"], rates["gyroscope.csv"]


def test_unequal_rate_fixture_reaches_same_50_hz_grid(tmp_path):
    fixture, acc_rate, gyro_rate = _write_unequal_rate_fixture(tmp_path / "fixture")
    config = PreprocessingConfig(minimum_segment_overlap_seconds=2.56)
    acc_frame = pd.read_csv(fixture / "accelerometer.csv")
    gyro_frame = pd.read_csv(fixture / "gyroscope.csv")
    acc = split_at_interpolation_gaps(
        split_sensor_segments(acc_frame, config)[0][0], config
    )[0]
    gyro = split_at_interpolation_gaps(
        split_sensor_segments(gyro_frame, config)[0][0], config
    )[0]
    timestamps, values, _, rejections = align_sensor_segments(
        acc, gyro, config
    )
    assert acc_rate != gyro_rate
    assert not rejections
    assert np.median(np.diff(timestamps)) == 20_000_000
    assert values.shape[1] == 6
    assert timestamps[0] == max(acc.timestamp[0], gyro.timestamp[0])
    assert timestamps[-1] <= min(acc.timestamp[-1], gyro.timestamp[-1])
    assert timestamps[0] == 0
    assert timestamps[0] % 20_000_000 == 0


def test_unequal_rate_alignment_stays_inside_source_intersection():
    config = PreprocessingConfig(minimum_segment_overlap_seconds=2.56)
    acc = _segment(100, offset=0.0, duration=4.0)
    gyro = _segment(80, offset=0.5, duration=4.0)
    timestamps, _, _, rejections = align_sensor_segments(acc, gyro, config)
    assert not rejections
    assert timestamps[0] >= gyro.timestamp[0]
    assert timestamps[-1] <= acc.timestamp[-1]
    assert timestamps[-1] <= gyro.timestamp[-1]


def test_unequal_rate_filter_state_resets_between_fragments():
    config = PreprocessingConfig(interpolation_gap_seconds=0.25)
    timestamps = np.r_[
        np.arange(0, 3, 0.01),
        np.arange(3.5, 6, 0.01),
    ]
    frame = pd.DataFrame(
        {
            "Creation_Time": np.rint(timestamps * 1_000_000_000).astype(np.int64),
            "x": np.sin(timestamps),
            "y": np.cos(timestamps),
            "z": timestamps,
            "gt": "walk",
        }
    )
    sessions, _ = split_sensor_segments(frame, config)
    fragments = split_at_interpolation_gaps(sessions[0], config)
    state = FilterState()
    for fragment in fragments:
        state.begin_segment(str(fragment.segment_id))
    assert len(fragments) == 2
    assert state.reset_count == 2
    assert fragments[0].timestamp[-1] < 3_000_000_000
    assert fragments[1].timestamp[0] >= 3_500_000_000


def test_unequal_rate_gap_blocks_cross_gap_windows(tmp_path):
    fixture, _, _ = _write_unequal_rate_fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    summary = preprocess_saved_fixture_streaming(fixture, output, chunk_size=37)
    assert summary["accepted_windows"] == 1
    assert summary["rejections"]["insufficient_intersection"] == 3
    with np.load(output / "windows-000000.npz") as shard:
        record = json.loads(shard["metadata"][0])
        assert record["source_segment_ids"] == ["0.0", "0.0"]
        assert record["end_timestamp_raw"] < 3_000_000_000


def test_unequal_rate_metadata_has_traceable_source_fields(tmp_path):
    fixture, _, _ = _write_unequal_rate_fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    preprocess_saved_fixture_streaming(fixture, output, chunk_size=37)
    with np.load(output / "windows-000000.npz") as shard:
        record = json.loads(shard["metadata"][0])
    assert record["source_segment_ids"] == ["0.0", "0.0"]
    assert record["start_timestamp_raw"] < record["end_timestamp_raw"]
    assert record["label"] == "walk"
    assert record["user"] == "synthetic"
    assert record["config_version"] == "har-streaming-fixture-v1"
    assert record["source_segment_ids"] == ["0.0", "0.0"]
    assert "start_timestamp_raw" in record
    assert "end_timestamp_raw" in record


def test_unequal_rate_shard_has_one_finite_window(tmp_path):
    fixture, _, _ = _write_unequal_rate_fixture(tmp_path / "fixture")
    output = tmp_path / "output"
    preprocess_saved_fixture_streaming(fixture, output, chunk_size=37)
    with np.load(output / "windows-000000.npz") as shard:
        assert shard["X"].shape == (1, 128, 6)
        assert np.isfinite(shard["X"]).all()
        assert shard["labels"].tolist() == ["walk"]


def test_stage2_validation_uses_selected_group_and_records_provenance(tmp_path):
    fixture, _, _ = _write_unequal_rate_fixture(tmp_path / "fixture")
    output = tmp_path / "stage2"
    summary = preprocess_stage2_validation(
        fixture / "accelerometer.csv",
        fixture / "gyroscope.csv",
        output,
        chunk_size=37,
    )
    assert summary["validation_only"] is True
    assert summary["accepted_windows"] == 1
    assert summary["pair_comparisons"] == 2
    assert summary["shard_count"] == 1
    with np.load(output / "windows-000000.npz") as shard:
        record = json.loads(shard["metadata"][0])
        assert shard["X"].shape == (1, 128, 6)
        assert np.isfinite(shard["X"]).all()
    assert record["source_segment_ids"] == ["0.0", "0.0"]
    assert record["source_traceability_ref"]
    assert record["config_version"] == "har-stage2-bounded-v1"
    assert record["end_timestamp_raw"] < 3_000_000_000


def test_selected_group_spooling_filters_before_repair_and_reports_progress(
    tmp_path, capsys
):
    fixture, _, _ = _write_unequal_rate_fixture(tmp_path / "fixture")
    spool = tmp_path / "spool"
    spool.mkdir()
    selected = {("synthetic", "unequal", "paired-1")}

    paths = _spool_repaired_stream(
        fixture / "accelerometer.csv",
        "acc",
        spool,
        None,
        chunk_size=37,
        rss_config=BoundedStage2Config(),
        selected_groups=selected,
    )

    assert len(paths) == 1
    spool_files = list(spool.glob("*.csv"))
    assert len(spool_files) == 1
    assert set(
        map(
            tuple,
            pd.read_csv(spool_files[0])[["User", "Model", "Device"]]
            .astype(str)
            .drop_duplicates()
            .to_numpy(),
        )
    ) == selected
    progress = capsys.readouterr().out
    assert "sensor=acc" in progress
    assert "chunks=" in progress
    assert "rows_read=" in progress
    assert "elapsed_seconds=" in progress
    assert "selected_rows=" in progress
    assert "rows_spooled=" in progress


def test_stage2_validation_rejects_sensor_label_disagreement(tmp_path):
    fixture, _, _ = _write_unequal_rate_fixture(tmp_path / "fixture")
    for filename in ("accelerometer.csv", "gyroscope.csv"):
        frame = pd.read_csv(fixture / filename)
        if filename == "gyroscope.csv":
            frame.loc[100:160, "gt"] = "sit"
        frame.to_csv(fixture / filename, index=False)
    output = tmp_path / "stage2"
    summary = preprocess_stage2_validation(
        fixture / "accelerometer.csv",
        fixture / "gyroscope.csv",
        output,
        chunk_size=37,
    )
    assert summary["accepted_windows"] == 0
    assert summary["rejections"]["sensor_label_disagreement"] >= 1


def test_stage2_validation_does_not_create_cross_user_split_leakage():
    splits = make_subject_splits(["u1", "u2", "u3", "u4"], seed=4786)
    by_split = {}
    for user, split in splits.items():
        by_split.setdefault(split, set()).add(user)
    assert sum(len(users) for users in by_split.values()) == len(splits)
    assert not (by_split["train"] & by_split["val"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["val"] & by_split["test"])


def test_interval_sweep_returns_only_overlapping_pairs():
    left = [_segment(100, duration=3), _segment(100, offset=10, duration=3)]
    right = [_segment(100, offset=1, duration=3), _segment(100, offset=20, duration=1)]
    pairs = interval_sweep_overlaps(left, right)
    assert len(pairs) == 1
    assert pairs[0][0] is left[0]
    assert pairs[0][1] is right[0]


def test_stage2_rejects_fragments_above_supported_filter_length(tmp_path):
    fixture = FIXTURE_DIRECTORY
    output = tmp_path / "stage2"
    summary = preprocess_stage2_validation(
        fixture / "accelerometer.csv",
        fixture / "gyroscope.csv",
        output,
        chunk_size=37,
        stage2_config=BoundedStage2Config(max_in_memory_fragment_samples=10),
    )
    assert summary["accepted_windows"] == 0
    assert summary["rejections"]["fragment_exceeds_supported_filter_length"] >= 1
    manifest = json.loads((output / "shard_manifest.json").read_text())
    assert manifest["shards"] == []
    with gzip.open(output / "repair_provenance.jsonl.gz", "rt", encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == 173


def test_stage2_rss_guard_fails_closed(tmp_path, monkeypatch):
    fixture, _, _ = _write_unequal_rate_fixture(tmp_path / "fixture")
    output = tmp_path / "stage2"
    monkeypatch.setattr(
        har_streaming,
        "process_rss_bytes",
        lambda: STAGE2_RSS_LIMIT_BYTES + 1,
    )
    with pytest.raises(MemoryError):
        preprocess_stage2_validation(
            fixture / "accelerometer.csv",
            fixture / "gyroscope.csv",
            output,
            selected_groups={("synthetic", "unequal", "paired-1")},
            chunk_size=37,
        )
    failure = json.loads((output / "stage2_failure.json").read_text())
    assert failure["status"] == "failed_closed"


def test_stage2_resumes_from_last_completed_group(tmp_path, monkeypatch):
    fixture, _, _ = _write_unequal_rate_fixture(tmp_path / "fixture")
    for filename in ("accelerometer.csv", "gyroscope.csv"):
        path = fixture / filename
        frame = pd.read_csv(path)
        duplicate = frame.copy()
        duplicate["User"] = "synthetic-2"
        duplicate["Device"] = "paired-2"
        pd.concat([frame, duplicate], ignore_index=True).to_csv(path, index=False)
    output = tmp_path / "stage2"
    original_sweep = har_streaming.interval_sweep_overlaps
    calls = {"count": 0}

    def interrupt_after_first_group(left, right):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("intentional interruption")
        return original_sweep(left, right)

    monkeypatch.setattr(har_streaming, "interval_sweep_overlaps", interrupt_after_first_group)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        preprocess_stage2_validation(
            fixture / "accelerometer.csv",
            fixture / "gyroscope.csv",
            output,
            chunk_size=37,
        )
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    assert checkpoint["completed_group"] == ["synthetic", "unequal", "paired-1"]
    monkeypatch.setattr(har_streaming, "interval_sweep_overlaps", original_sweep)
    summary = preprocess_stage2_validation(
        fixture / "accelerometer.csv",
        fixture / "gyroscope.csv",
        output,
        chunk_size=37,
    )
    assert summary["accepted_windows"] == 2
    manifest = json.loads((output / "shard_manifest.json").read_text())
    assert sum(item["window_count"] for item in manifest["shards"]) == 2


def test_stage2_checkpoint_without_manifest_fails_closed(tmp_path):
    output = tmp_path / "stage2"
    output.mkdir()
    (output / "checkpoint.json").write_text(
        json.dumps(
            {
                "input_position": 0,
                "last_committed_sequence": 0,
                "last_committed_sha256": "missing",
                "completed_group_index": 0,
                "completed_group": ["a", "m", "d"],
            }
        )
    )
    with pytest.raises(RuntimeError, match="manifest"):
        preprocess_stage2_validation(
            FIXTURE_DIRECTORY / "accelerometer.csv",
            FIXTURE_DIRECTORY / "gyroscope.csv",
            output,
        )


def test_stage2_checkpoint_is_not_advanced_when_window_write_fails(tmp_path, monkeypatch):
    fixture, _, _ = _write_unequal_rate_fixture(tmp_path / "fixture")
    output = tmp_path / "stage2"
    original_append = har_streaming.CompressedShardWriter.append

    def fail_append(self, values, label, metadata):
        raise OSError("injected window write failure")

    monkeypatch.setattr(har_streaming.CompressedShardWriter, "append", fail_append)
    with pytest.raises(OSError, match="window write failure"):
        preprocess_stage2_validation(
            fixture / "accelerometer.csv",
            fixture / "gyroscope.csv",
            output,
            chunk_size=37,
        )
    assert not (output / "checkpoint.json").exists()
    monkeypatch.setattr(har_streaming.CompressedShardWriter, "append", original_append)


def test_bounded_stage2_matches_v2_oracle(tmp_path):
    output = tmp_path / "stage2"
    summary = preprocess_stage2_validation(
        FIXTURE_DIRECTORY / "accelerometer.csv",
        FIXTURE_DIRECTORY / "gyroscope.csv",
        output,
        chunk_size=101,
    )
    assert summary["accepted_windows"] == 2
    with np.load(output / "windows-000000.npz") as shard:
        expected_x = np.load(V2_OUTPUT / "X_raw.npy")
        expected_metadata = pd.read_csv(V2_OUTPUT / "metadata.csv")
        records = [json.loads(value) for value in shard["metadata"]]
        np.testing.assert_array_equal(shard["X"], expected_x)
        assert shard["labels"].tolist() == expected_metadata["label"].tolist()
    assert [
        (item["start_timestamp_raw"], item["end_timestamp_raw"])
        for item in records
    ] == list(
        zip(
            expected_metadata["start_timestamp_raw"],
            expected_metadata["end_timestamp_raw"],
        )
    )
    assert [item["source_segment_ids"] for item in records] == [
        ["2.0", "1.0"], ["2.1", "1.1"]
    ]


def test_resample_and_window_stage2_emits_contract_metadata(tmp_path):
    output = tmp_path / "windows"
    summary = resample_and_window_stage2(
        FIXTURE_DIRECTORY / "accelerometer.csv",
        FIXTURE_DIRECTORY / "gyroscope.csv",
        output,
        chunk_size=101,
    )
    assert summary["accepted_windows"] == 2
    with np.load(output / "windows-000000.npz") as shard:
        records = [json.loads(value) for value in shard["metadata"]]
        assert shard["X"].shape == (2, 128, 6)
        assert all(
            record["target_sample_count"] == window.shape[0] == 128
            for record, window in zip(records, shard["X"])
        )
    for record in records:
        assert record["input_shard"] == {
            "acc": "accelerometer.csv",
            "gyro": "gyroscope.csv",
        }
        assert record["target_sample_count"] >= 128
        assert record["target_sample_count"] == 128
        assert record["aligned_target_grid_sample_count"] >= 128
        assert record["source_sample_counts"]["acc"] >= 128
        assert record["source_sample_counts"]["gyro"] >= 128
        assert record["resampling"]["target_hz"] == 50.0
        assert record["resampling"]["target_step_raw"] == 20_000_000
        assert record["resampling"]["filter_semantics"] == (
            "complete-fragment-zero-phase"
        )
