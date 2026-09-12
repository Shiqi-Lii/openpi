#!/usr/bin/env python3
"""Standalone replay of one 16D NZ100 LeRobot episode.

Dry run and video playback import no ROS 2, ``robot_client``, or external
YSRobot Python SDK. ``--execute`` publishes joint trajectories through ROS 2
and controls grippers through the robot HTTP API; it remains opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np


DEFAULT_DATASET = Path("/home/pc/VLA/lerobot_data_collection/data_lerobot")
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
ACTION_DIM = 16


def main() -> None:
    args = _parse_args()
    actions, timestamps = _load_episode_actions(args.dataset, args.episode, args.action_key)
    source_fps = float(args.fps) if args.fps is not None else _infer_fps(args.dataset, timestamps)
    frame_indices = np.arange(len(actions))[args.start : args.end : args.stride]
    actions = actions[args.start : args.end : args.stride]
    timestamps = timestamps[args.start : args.end : args.stride]
    if len(actions) == 0:
        raise ValueError("No actions selected for replay")

    fps = source_fps / args.stride
    period_s = 1.0 / fps if fps > 0 else 0.0
    video_keys = _select_video_keys(args.dataset, args.video_keys, args.video_key) if args.show_video else []

    print(
        f"NZ100 replay: dataset={args.dataset}, episode={args.episode:06d}, "
        f"actions={len(actions)}, fps={fps:.3f}, execute={args.execute}"
    )
    print(f"first action: {_format_action(actions[0])}")
    if video_keys:
        print(f"video views: {video_keys}")

    robot = _StandaloneRobot(args) if args.execute else None
    videos = _MultiViewVideo(args.dataset, args.episode, video_keys) if video_keys else None
    try:
        if robot is not None:
            robot.connect()
            if args.home:
                robot.home()
            else:
                print("Skipping home before replay.")
        _replay(actions, frame_indices, robot, videos, period_s, args.max_steps, args.print_every)
    finally:
        if videos is not None:
            videos.close()
        if robot is not None:
            robot.close()


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = config_parser.parse_known_args()
    config = _load_config(config_args.config)

    parser = argparse.ArgumentParser(description="Standalone NZ100 LeRobot episode replay")
    parser.add_argument("--config", type=Path, default=config_args.config, help="Replay YAML configuration")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--video-keys", nargs="+", default=None, help="Views to show; default discovers all videos")
    parser.add_argument("--video-key", default=None, help="Deprecated single-view alias")

    parser.add_argument("--execute", action="store_true", help="Send actions to the real robot through HTTP")
    parser.add_argument("--home", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot-host", default="192.168.2.123")
    parser.add_argument("--robot-port", type=int, default=5010)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--login-level", default="L4")
    parser.add_argument("--login-pin", default="admin")
    parser.add_argument("--point-time-from-start", type=float, default=0.1)
    parser.add_argument("--home-time-from-start", type=float, default=2.0)
    parser.add_argument("--left-trajectory-topic", default="/arm_left_controller/joint_trajectory")
    parser.add_argument("--right-trajectory-topic", default="/arm_right_controller/joint_trajectory")
    parser.add_argument("--left-joint-names", nargs=7, default=[f"left_joint{i}" for i in range(1, 8)])
    parser.add_argument("--right-joint-names", nargs=7, default=[f"right_joint{i}" for i in range(1, 8)])
    parser.add_argument("--left-gripper-register", type=int, default=9661)
    parser.add_argument("--right-gripper-register", type=int, default=9662)
    parser.add_argument("--left-home", type=float, nargs=7, default=[0.28, 0.17, 0.09, 1.83, 1.75, -0.09, 0.0])
    parser.add_argument("--right-home", type=float, nargs=7, default=[-0.28, 0.17, -0.09, 1.83, -1.75, -0.09, 0.0])
    parser.set_defaults(**config)
    args = parser.parse_args()
    if args.stride <= 0:
        parser.error("--stride must be positive")
    return args


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Replay config not found: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to read the replay configuration") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Replay config must be a YAML mapping: {path}")

    allowed = {
        "dataset",
        "episode",
        "action_key",
        "fps",
        "start",
        "end",
        "stride",
        "max_steps",
        "print_every",
        "show_video",
        "video_keys",
        "execute",
        "home",
        "robot_host",
        "robot_port",
        "timeout_s",
        "login_level",
        "login_pin",
        "point_time_from_start",
        "home_time_from_start",
        "left_trajectory_topic",
        "right_trajectory_topic",
        "left_joint_names",
        "right_joint_names",
        "left_gripper_register",
        "right_gripper_register",
        "left_home",
        "right_home",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown replay config keys: {unknown}")
    if "dataset" in data:
        data["dataset"] = Path(data["dataset"]).expanduser()
    return data


def _episode_path(dataset: Path, episode: int) -> Path:
    path = dataset / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return path


def _load_episode_actions(dataset: Path, episode: int, action_key: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas and pyarrow to read LeRobot parquet files") from exc
    path = _episode_path(dataset, episode)
    frame = pd.read_parquet(path)
    if action_key not in frame.columns:
        raise KeyError(f"Column {action_key!r} not found; columns={list(frame.columns)}")
    actions = np.stack([np.asarray(x, dtype=np.float32) for x in frame[action_key]], axis=0)
    if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
        raise ValueError(f"Expected action shape (T, >= {ACTION_DIM}), got {actions.shape}")
    if actions.shape[1] > ACTION_DIM:
        print(
            f"Dataset action has {actions.shape[1]} dimensions; "
            f"replay uses the first {ACTION_DIM} joint/gripper dimensions and ignores the rest."
        )
        actions = actions[:, :ACTION_DIM]
    timestamps = (
        np.asarray(frame["timestamp"], dtype=np.float64)
        if "timestamp" in frame.columns
        else np.arange(len(actions), dtype=np.float64)
    )
    return actions, timestamps


def _infer_fps(dataset: Path, timestamps: np.ndarray) -> float:
    info_path = dataset / "meta" / "info.json"
    if info_path.exists():
        try:
            fps = float(json.loads(info_path.read_text(encoding="utf-8")).get("fps", 0))
            if fps > 0:
                return fps
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    if len(timestamps) > 1:
        positive_dt = np.diff(timestamps)
        positive_dt = positive_dt[positive_dt > 0]
        if len(positive_dt):
            return 1.0 / float(np.median(positive_dt))
    return 30.0


def _discover_video_keys(dataset: Path) -> list[str]:
    videos = dataset / "videos"
    if not videos.exists():
        return []
    keys: set[str] = set()
    for chunk in videos.glob("chunk-*"):
        for path in chunk.rglob("episode_*.mp4"):
            keys.add(path.parent.relative_to(chunk).as_posix())
    return sorted(keys)


def _select_video_keys(dataset: Path, keys: list[str] | None, legacy_key: str | None) -> list[str]:
    selected = list(keys or ([legacy_key] if legacy_key else _discover_video_keys(dataset)))
    if not selected:
        raise FileNotFoundError(f"No episode videos found under {dataset / 'videos'}")
    return selected


def _video_path(dataset: Path, episode: int, key: str) -> Path:
    return dataset / "videos" / f"chunk-{episode // 1000:03d}" / key / f"episode_{episode:06d}.mp4"


class _MultiViewVideo:
    def __init__(self, dataset: Path, episode: int, keys: list[str]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install opencv-python to use --show-video") from exc
        self.cv2 = cv2
        self.window = f"NZ100 replay episode_{episode:06d}"
        self.keys = keys
        self.captures = []
        for key in keys:
            path = _video_path(dataset, episode, key)
            if not path.exists():
                raise FileNotFoundError(f"Episode video not found: {path}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open episode video: {path}")
            self.captures.append(cap)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    def show(self, frame_index: int) -> bool:
        frames = []
        for key, cap in zip(self.keys, self.captures, strict=True):
            cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((360, 480, 3), dtype=np.uint8)
                self.cv2.putText(frame, "frame unavailable", (20, 180), 0, 0.7, (0, 0, 255), 2)
            frame = _fit_view(self.cv2, frame, 480, 360)
            self.cv2.putText(frame, key, (10, 28), 0, 0.7, (255, 255, 255), 2)
            frames.append(frame)
        columns = min(2, len(frames))
        rows = math.ceil(len(frames) / columns)
        blank = np.zeros_like(frames[0])
        frames.extend([blank] * (rows * columns - len(frames)))
        canvas = np.vstack([np.hstack(frames[i : i + columns]) for i in range(0, len(frames), columns)])
        self.cv2.imshow(self.window, canvas)
        return (self.cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        for cap in self.captures:
            cap.release()
        self.cv2.destroyWindow(self.window)


def _fit_view(cv2, image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _replay(actions, frame_indices, robot, videos, period_s, max_steps, print_every) -> None:
    limit = len(actions) if max_steps <= 0 else min(len(actions), max_steps)
    next_tick = time.monotonic()
    completed = 0
    for step in range(limit):
        if videos is not None and not videos.show(int(frame_indices[step])):
            print("'q' pressed; stopping replay.")
            break
        if print_every > 0 and step % print_every == 0:
            print(f"Replay action[{step}/{limit}]: {_format_action(actions[step])}")
        if robot is not None:
            robot.apply(actions[step])
        completed += 1
        next_tick += period_s
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()
    print(f"Replay finished: {completed} actions")


def _gripper(value: float) -> int:
    return 2 if float(value) >= 1.5 else 1


def _format_action(action: np.ndarray) -> str:
    return (
        f"left={np.array2string(action[:7], precision=3)}, left_gripper={_gripper(action[7])}, "
        f"right={np.array2string(action[8:15], precision=3)}, right_gripper={_gripper(action[15])}"
    )


class _StandaloneRobot:
    """Standalone ROS trajectory publishers plus YSRobot Modbus HTTP."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base = f"http://{args.robot_host}:{args.robot_port}"
        self.token: str | None = None
        self.last_grippers: tuple[int, int] | None = None
        self.rclpy = None
        self.node = None
        self.left_publisher = None
        self.right_publisher = None

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urlrequest.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urlrequest.urlopen(req, timeout=self.args.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Robot HTTP {method} {path} failed: {exc}") from exc

    @staticmethod
    def _check(result: dict, operation: str) -> None:
        if result.get("status") not in ("success", "1", "true", True, 200):
            raise RuntimeError(f"{operation} failed: {result.get('message', result)}")

    def connect(self) -> None:
        try:
            import rclpy
            from trajectory_msgs.msg import JointTrajectory
        except ImportError as exc:
            raise RuntimeError("ROS2 is required only for --execute; source /opt/ros/humble/setup.bash") from exc
        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("nz100_standalone_replay")
        self.left_publisher = self.node.create_publisher(
            JointTrajectory, self.args.left_trajectory_topic, 10
        )
        self.right_publisher = self.node.create_publisher(
            JointTrajectory, self.args.right_trajectory_topic, 10
        )

        pin_hash = hashlib.sha256(self.args.login_pin.encode("utf-8")).hexdigest()
        result = self._call("POST", "/api/auth/login", {"level": self.args.login_level, "pinHash": pin_hash})
        self._check(result, "Robot login")
        self.token = (result.get("data") or {}).get("token")
        if not self.token:
            raise RuntimeError("Robot login succeeded without a token")
        print(
            f"Robot control ready: joints={self.args.left_trajectory_topic},"
            f"{self.args.right_trajectory_topic}; grippers={self.args.robot_host}:{self.args.robot_port}"
        )
        time.sleep(0.2)  # Allow DDS publisher/subscriber discovery before the first command.

    def home(self) -> None:
        print("Moving both arms to replay home pose.")
        self._publish_joints(
            self.left_publisher,
            self.args.left_joint_names,
            self.args.left_home,
            duration_s=self.args.home_time_from_start,
        )
        self._publish_joints(
            self.right_publisher,
            self.args.right_joint_names,
            self.args.right_home,
            duration_s=self.args.home_time_from_start,
        )
        self._write_gripper(self.args.left_gripper_register, 1)
        self._write_gripper(self.args.right_gripper_register, 1)
        self.last_grippers = (1, 1)
        time.sleep(float(self.args.home_time_from_start))

    def apply(self, action: np.ndarray) -> None:
        self._publish_joints(self.left_publisher, self.args.left_joint_names, action[:7])
        self._publish_joints(self.right_publisher, self.args.right_joint_names, action[8:15])
        grippers = (_gripper(action[7]), _gripper(action[15]))
        if self.last_grippers is None or grippers[0] != self.last_grippers[0]:
            self._write_gripper(self.args.left_gripper_register, grippers[0])
        if self.last_grippers is None or grippers[1] != self.last_grippers[1]:
            self._write_gripper(self.args.right_gripper_register, grippers[1])
        self.last_grippers = grippers

    def _publish_joints(self, publisher, joint_names, positions, *, duration_s: float | None = None) -> None:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        duration_s = float(self.args.point_time_from_start if duration_s is None else duration_s)
        if duration_s <= 0:
            raise ValueError(f"Trajectory duration must be positive, got {duration_s}")
        msg = JointTrajectory()
        msg.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start = Duration(
            sec=int(duration_s),
            nanosec=int((duration_s % 1.0) * 1e9),
        )
        msg.points = [point]
        publisher.publish(msg)

    def _write_gripper(self, address: int, value: int) -> None:
        result = self._call("POST", "/api/modbus_slave/write", {"address": address, "value": value})
        self._check(result, f"Modbus write {address}")

    def close(self) -> None:
        self.token = None
        if self.node is not None:
            self.node.destroy_node()
        if self.rclpy is not None and self.rclpy.ok():
            self.rclpy.shutdown()
        print("Robot connection closed")


if __name__ == "__main__":
    main()
