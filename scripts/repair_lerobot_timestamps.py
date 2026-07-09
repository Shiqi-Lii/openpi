#!/usr/bin/env python3
"""Repair LeRobot parquet timestamps to be frame_index / fps per episode.

LeRobot v2.1 checks that consecutive timestamps inside each episode differ by
1 / fps seconds. Some converters accidentally store absolute/repeated wall-clock
timestamps, which makes local dataset loading fail before training starts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

import numpy as np


def _data_path(dataset_root: pathlib.Path, info: dict, episode_index: int) -> pathlib.Path:
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    return dataset_root / data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)


def _repair_with_pyarrow(parquet_path: pathlib.Path, fps: int, dry_run: bool) -> tuple[bool, str]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    names = table.column_names
    if "timestamp" not in names or "frame_index" not in names:
        return False, "missing timestamp/frame_index"

    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.float64)
    expected = frame_index / float(fps)
    current = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    changed = not np.allclose(current, expected, rtol=0.0, atol=1e-6)

    if changed and not dry_run:
        timestamp = pa.array(expected.astype(np.float64))
        table = table.set_column(names.index("timestamp"), "timestamp", timestamp)
        with tempfile.NamedTemporaryFile(dir=parquet_path.parent, suffix=".parquet", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            pq.write_table(table, tmp_path)
            tmp_path.replace(parquet_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    first = float(current[0]) if len(current) else 0.0
    second = float(current[1]) if len(current) > 1 else first
    return changed, f"old first two timestamps: {first}, {second}"


def _repair_with_pandas(parquet_path: pathlib.Path, fps: int, dry_run: bool) -> tuple[bool, str]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "timestamp" not in df.columns or "frame_index" not in df.columns:
        return False, "missing timestamp/frame_index"

    expected = df["frame_index"].to_numpy(dtype=np.float64) / float(fps)
    current = df["timestamp"].to_numpy(dtype=np.float64)
    changed = not np.allclose(current, expected, rtol=0.0, atol=1e-6)

    if changed and not dry_run:
        df["timestamp"] = expected
        with tempfile.NamedTemporaryFile(dir=parquet_path.parent, suffix=".parquet", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            df.to_parquet(tmp_path, index=False)
            tmp_path.replace(parquet_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    first = float(current[0]) if len(current) else 0.0
    second = float(current[1]) if len(current) > 1 else first
    return changed, f"old first two timestamps: {first}, {second}"


def repair(dataset_root: pathlib.Path, dry_run: bool = False) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    with (dataset_root / "meta/info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)
    with (dataset_root / "meta/episodes.jsonl").open("r", encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    fps = int(info["fps"])
    changed_count = 0
    for episode in sorted(episodes, key=lambda item: item["episode_index"]):
        episode_index = int(episode["episode_index"])
        parquet_path = _data_path(dataset_root, info, episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)

        try:
            changed, message = _repair_with_pyarrow(parquet_path, fps, dry_run)
        except ModuleNotFoundError:
            changed, message = _repair_with_pandas(parquet_path, fps, dry_run)

        if changed:
            changed_count += 1
            action = "would repair" if dry_run else "repaired"
            print(f"episode {episode_index:06d}: {action}; {message}")

    print(f"{'Would repair' if dry_run else 'Repaired'} {changed_count}/{len(episodes)} parquet files.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repair(args.dataset_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
