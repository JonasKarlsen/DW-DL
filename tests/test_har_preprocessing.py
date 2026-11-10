import json

import numpy as np
import pandas as pd

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
    ChunkRepairState,
    CompressedShardWriter,
    FilterState,
    WindowOverlapBuffer,
)


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
