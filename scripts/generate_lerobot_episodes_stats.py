#!/usr/bin/env python3
"""Generate LeRobot v2.1 meta/episodes_stats.jsonl for a local dataset.

Some converted LeRobot datasets contain meta/stats.json but miss
meta/episodes_stats.jsonl. LeRobot v2.1 expects per-episode stats when loading
local datasets, so OpenPI training fails before it can compute its own
normalization stats.

This script computes per-episode stats from the parquet data files. Video stats
are intentionally skipped: OpenPI does not use image/video normalization from
LeRobot metadata, and skipping them keeps this repair fast and dependency-light.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np


STATS = ("min", "max", "mean", "std")


def _to_array(values: list[Any], key: str) -> np.ndarray:
    if len(values) == 0:
        raise ValueError(f"Cannot compute stats for empty column {key!r}")

    first = values[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack([np.asarray(item) for item in values])

    return np.asarray(values)


def _feature_stats(array: np.ndarray) -> dict[str, list[Any]]:
    return {
        "min": np.min(array, axis=0, keepdims=array.ndim == 1).tolist(),
        "max": np.max(array, axis=0, keepdims=array.ndim == 1).tolist(),
        "mean": np.mean(array, axis=0, keepdims=array.ndim == 1).tolist(),
        "std": np.std(array, axis=0, keepdims=array.ndim == 1).tolist(),
        "count": [int(len(array))],
    }


def _data_path(dataset_root: pathlib.Path, info: dict, episode_index: int) -> pathlib.Path:
    data_path = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    return dataset_root / data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)


def _read_parquet_columns(parquet_path: pathlib.Path, columns: list[str]) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        pq = None

    if pq is not None:
        table = pq.read_table(parquet_path, columns=columns)
        return {name: table[name].to_pylist() for name in table.column_names}

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading parquet requires pyarrow or pandas. Run this script inside the OpenPI "
            "training environment, for example via scripts/train_nz100.sh."
        ) from exc

    df = pd.read_parquet(parquet_path, columns=columns)
    return {name: df[name].tolist() for name in df.columns}


def generate(dataset_root: pathlib.Path, overwrite: bool = False) -> pathlib.Path:
    dataset_root = dataset_root.expanduser().resolve()
    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    output_path = meta_dir / "episodes_stats.jsonl"

    if output_path.exists() and not overwrite:
        print(f"{output_path} already exists; use --overwrite to regenerate it.")
        return output_path

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    with episodes_path.open("r", encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    features = info["features"]
    stat_keys = [
        key
        for key, feature in features.items()
        if feature.get("dtype") not in {"image", "video", "string"} and key != "index"
    ]

    rows = []
    for episode in sorted(episodes, key=lambda item: item["episode_index"]):
        episode_index = int(episode["episode_index"])
        parquet_path = _data_path(dataset_root, info, episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)

        columns = [key for key in stat_keys if key in features]
        data = _read_parquet_columns(parquet_path, columns)
        stats = {}
        for key in stat_keys:
            if key not in data:
                continue
            stats[key] = _feature_stats(_to_array(data[key], key))

        rows.append({"episode_index": episode_index, "stats": stats})

    tmp_path = output_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    tmp_path.replace(output_path)
    print(f"Wrote {len(rows)} episode stats to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=pathlib.Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate(args.dataset_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
