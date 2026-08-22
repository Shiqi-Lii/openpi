#!/usr/bin/env python3
"""Replay one LeRobot episode on NZ100.

The expected LeRobot action layout is the current NZ100 16D order:

    0:7     left arm joints
    7       left gripper, PLC convention 1=open, 2=closed
    8:15    right arm joints
    15      right gripper, PLC convention 1=open, 2=closed

By default this script is a dry-run. Pass ``--execute`` to actually publish
joint trajectories and Modbus gripper commands through ``robot_client``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from robot_client.config import load_app_config
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.runners.common import format_action
from robot_client.state_builder import discretize_plc_grippers
from robot_client.state_builder import split_action


DEFAULT_DATASET = Path("/home/pc/VLA/lerobot_data_collection/data_lerobot")
DEFAULT_CONFIG = Path("robot_client/configs/nz100_client.yaml")


def main() -> None:
    args = _parse_args()
    actions, timestamps = _load_episode_actions(
        args.dataset,
        episode_index=args.episode,
        action_key=args.action_key,
    )
    actions = _slice_actions(actions, start=args.start, end=args.end, stride=args.stride)
    timestamps = _slice_actions(timestamps, start=args.start, end=args.end, stride=args.stride)
    if len(actions) == 0:
        raise ValueError("No actions selected for replay")

    fps = float(args.fps) if args.fps is not None else _infer_fps(args.dataset, timestamps)
    period_s = 1.0 / fps if fps > 0 else 0.0

    print(
        "NZ100 replay: "
        f"dataset={args.dataset}, episode={args.episode:06d}, actions={len(actions)}, "
        f"fps={fps:.3f}, dry_run={not args.execute}"
    )
    print(f"first action: {format_action(discretize_plc_grippers(split_action(actions[0])))}")

    ros_io = None
    try:
        if args.execute:
            app_config = load_app_config(args.config)
            ros_io = NZ100Ros2IO(app_config.ros2)
            ros_io.connect()
            if args.home:
                ros_io.move_to_home()
            else:
                print("Skipping home before replay.")

        _replay_actions(
            actions,
            ros_io=ros_io,
            dataset=args.dataset,
            episode_index=args.episode,
            video_key=args.video_key,
            show_video=args.show_video,
            period_s=period_s,
            execute=args.execute,
            max_steps=args.max_steps,
            print_every=args.print_every,
        )
    finally:
        if ros_io is not None:
            ros_io.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one 16D NZ100 LeRobot episode")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="robot_client YAML config")
    parser.add_argument("--action-key", default="action", help="Parquet column to replay")
    parser.add_argument("--fps", type=float, default=None, help="Replay FPS; default uses meta/info.json or timestamps")
    parser.add_argument("--start", type=int, default=0, help="Start frame index inside the episode")
    parser.add_argument("--end", type=int, default=None, help="End frame index, exclusive")
    parser.add_argument("--stride", type=int, default=1, help="Replay every Nth frame")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum replay steps; 0 means no limit")
    parser.add_argument("--print-every", type=int, default=10, help="Print every N replayed actions")
    parser.add_argument("--execute", action="store_true", help="Actually send commands to the robot")
    parser.add_argument("--home", action=argparse.BooleanOptionalAction, default=True, help="Home robot before replay")
    parser.add_argument("--show-video", action="store_true", help="Open a window and play the episode video during replay")
    parser.add_argument("--video-key", default="observation.images.top", help="LeRobot video key to display")
    return parser.parse_args()


def _episode_path(dataset: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    path = dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return path


def _load_episode_actions(dataset: Path, *, episode_index: int, action_key: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas/pyarrow are required to read LeRobot parquet files") from exc

    episode_path = _episode_path(dataset, episode_index)
    df = pd.read_parquet(episode_path)
    if action_key not in df.columns:
        raise KeyError(f"Column {action_key!r} not found in {episode_path}; columns={list(df.columns)}")

    actions = np.stack([np.asarray(value, dtype=np.float32) for value in df[action_key].to_list()], axis=0)
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise ValueError(f"Expected action shape (T, 16), got {actions.shape}")

    if "timestamp" in df.columns:
        timestamps = np.asarray(df["timestamp"].to_numpy(), dtype=np.float64)
    else:
        timestamps = np.arange(len(actions), dtype=np.float64)
    return actions, timestamps


def _slice_actions(values: np.ndarray, *, start: int, end: int | None, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return values[start:end:stride]


def _infer_fps(dataset: Path, timestamps: np.ndarray) -> float:
    info_path = dataset / "meta" / "info.json"
    if info_path.exists():
        try:
            import json

            info = json.loads(info_path.read_text(encoding="utf-8"))
            fps = float(info.get("fps", 0.0))
            if fps > 0:
                return fps
        except Exception:
            pass
    if len(timestamps) >= 2:
        dt = float(np.median(np.diff(timestamps)))
        if dt > 0:
            return 1.0 / dt
    return 30.0


def _replay_actions(
    actions: np.ndarray,
    *,
    ros_io: NZ100Ros2IO | None,
    dataset: Path,
    episode_index: int,
    video_key: str,
    show_video: bool,
    period_s: float,
    execute: bool,
    max_steps: int,
    print_every: int,
) -> None:
    video = _EpisodeVideo(dataset, episode_index=episode_index, video_key=video_key) if show_video else None
    next_tick = time.monotonic()
    limit = len(actions) if max_steps <= 0 else min(len(actions), int(max_steps))
    executed = 0
    try:
        for step, raw_action in enumerate(actions[:limit]):
            if video is not None and not video.show(step):
                print("Video window closed or 'q' pressed; stopping replay.")
                break

            action = discretize_plc_grippers(split_action(raw_action))
            if print_every > 0 and step % print_every == 0:
                print(f"Replay action[{step}/{limit}]: {format_action(action)}")
            if execute:
                ros_io.apply_action(action)

            executed += 1
            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
        print(f"Replay finished: {executed} actions")
    finally:
        if video is not None:
            video.close()


def _video_path(dataset: Path, *, episode_index: int, video_key: str) -> Path:
    chunk = episode_index // 1000
    return dataset / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


class _EpisodeVideo:
    """Tiny OpenCV video window for episode replay."""

    def __init__(self, dataset: Path, *, episode_index: int, video_key: str) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for --show-video") from exc

        self._cv2 = cv2
        self._path = _video_path(dataset, episode_index=episode_index, video_key=video_key)
        if not self._path.exists():
            raise FileNotFoundError(f"Episode video not found: {self._path}")
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open episode video: {self._path}")
        self._window = f"NZ100 replay {video_key} episode_{episode_index:06d}"
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        print(f"Showing replay video: {self._path}")

    def show(self, frame_index: int) -> bool:
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self._cap.read()
        if not ok:
            print(f"Warning: failed to read video frame {frame_index} from {self._path}")
            return True
        self._cv2.imshow(self._window, frame)
        key = self._cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        self._cap.release()
        self._cv2.destroyWindow(self._window)


if __name__ == "__main__":
    main()
