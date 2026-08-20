#!/usr/bin/env python3
"""Run VLA continuously and interrupt it with a perception-triggered grasp slot.

The VLA client stays alive in this process. A monitor thread polls:
1. YOLO bottle center xyz from an HTTP endpoint;
2. left TCP xyz from YSRobot SDK ``motion.get_pose``.

When the distance is below a threshold, the monitor requests a pause. The RTC
runner pauses on the next action boundary, runs a placeholder grasp policy, then
resumes VLA from a fresh observation/action chunk.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from robot_client.config import AppConfig
from robot_client.config import ClientConfig
from robot_client.config import load_app_config
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.runners import async_queue
from robot_client.runners import legato
from robot_client.runners import rtc_guidance
from robot_client.runners import sync_chunk
from robot_client.runtime_control import RuntimeControl


@dataclasses.dataclass(frozen=True)
class InterruptionConfig:
    """Perception-triggered interruption settings."""

    robot_client_config: str = "robot_client/configs/nz100_client.yaml"
    yolo_url: str | None = None
    yolo_timeout_s: float = 0.05
    poll_hz: float = 20.0
    distance_threshold_m: float = 0.05
    cooldown_s: float = 2.0
    max_interruptions: int = 1
    left_pose_frame: str | None = None
    grasp_move_vel: float = 5.0
    grasp_move_acc: float = 20.0
    grasp_move_planner: str = "pilz"
    handoff_hold_enabled: bool = True
    handoff_hold_s: float = 0.1
    lift_mm: float = 60.0
    no_lift: bool = False
    gripper_register: int = 9661
    gripper_close_value: int = 2
    gripper_settle_s: float = 1.0
    mock_yolo_xyz: tuple[float, float, float] | None = None


def load_interruption_config(path: Path | None) -> InterruptionConfig:
    if path is None:
        return InterruptionConfig()
    with path.open("r", encoding="utf-8") as file:
        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required for YAML configs. Install with: pip install pyyaml") from exc
            data = yaml.safe_load(file)
        else:
            data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return InterruptionConfig(**_filter_kwargs(InterruptionConfig, data))


def _filter_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    valid = {field.name for field in dataclasses.fields(cls)}
    result = {}
    for key, value in data.items():
        if key not in valid:
            continue
        if key == "mock_yolo_xyz" and value is not None:
            value = tuple(float(v) for v in value)
        result[key] = value
    return result


@dataclasses.dataclass(frozen=True)
class VisionTarget:
    pose_7d_m: list[float]


class YoloHttpClient:
    """HTTP client for the bottle target service."""

    def __init__(
        self,
        url: str | None,
        *,
        timeout_s: float,
        mock_xyz: tuple[float, float, float] | None,
    ) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._mock_xyz = mock_xyz

    def get_bottle_xyz(self) -> tuple[float, float, float] | None:
        target = self.get_bottle_target()
        if target is None:
            return None
        return float(target.pose_7d_m[0]), float(target.pose_7d_m[1]), float(target.pose_7d_m[2])

    def get_bottle_grasp_pose(self) -> list[float] | None:
        target = self.get_bottle_target()
        return None if target is None else target.pose_7d_m

    def get_bottle_target(self) -> VisionTarget | None:
        if self._mock_xyz is not None:
            return VisionTarget([*self._mock_xyz, 0.0, 0.0, 0.0, 1.0])
        if not self._url:
            return None
        try:
            payload = _http_get_json(self._url, timeout_s=self._timeout_s)
        except Exception as exc:
            print(f"YOLO target request failed: {exc}")
            return None
        pose = _extract_pose_7d(payload)
        if pose is None:
            return None
        return VisionTarget(pose)

    def get_lift_pose(self, target: VisionTarget, *, lift_mm: float) -> list[float]:
        """Lift along robot-default/base +Z without using grasp/latest."""

        lift_pose = [
            target.pose_7d_m[0],
            target.pose_7d_m[1],
            target.pose_7d_m[2] + float(lift_mm) / 1000.0,
            *target.pose_7d_m[3:7],
        ]
        print(f"Lift target pose_7d_m: {lift_pose} from base +Z, lift_mm={lift_mm}")
        return lift_pose


class LeftTcpPoseClient:
    """Reads left TCP xyz with YSRobot SDK."""

    def __init__(self, app_config: AppConfig, *, frame: str | None) -> None:
        sdk_path = str(Path(app_config.ros2.ysrobot_sdk_path).expanduser())
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        from ysrobot import ArmType, RobotClient

        self._arm_type = ArmType
        self._frame = frame
        self._robot = RobotClient(
            app_config.ros2.ysrobot_host,
            port=int(app_config.ros2.ysrobot_port),
            timeout_ms=int(app_config.ros2.ysrobot_timeout_ms),
        )
        result = self._robot.login(app_config.ros2.ysrobot_login_level, app_config.ros2.ysrobot_login_pin)
        print("Monitor YSRobot login:", result.success, result.message)
        if not result:
            raise RuntimeError(f"YSRobot login failed: {result.message}")
        result = self._robot.connect()
        print("Monitor YSRobot connect:", result.success, result.message)
        if not result:
            raise RuntimeError(f"YSRobot connect failed: {result.message}")

    def get_left_xyz(self) -> tuple[float, float, float]:
        pose = self._robot.motion.get_pose(self._arm_type.Left, frame=self._frame)
        return float(pose.x), float(pose.y), float(pose.z)

    def get_left_pose_7d(self) -> list[float]:
        pose = self._robot.motion.get_pose(self._arm_type.Left, frame=self._frame)
        return [
            float(pose.x),
            float(pose.y),
            float(pose.z),
            float(pose.qx),
            float(pose.qy),
            float(pose.qz),
            float(pose.qw),
        ]

    def with_current_left_orientation(self, pose_7d_m: list[float]) -> list[float]:
        if len(pose_7d_m) != 7:
            raise ValueError(f"Expected 7D target pose, got {len(pose_7d_m)} values: {pose_7d_m}")
        current = self.get_left_pose_7d()
        merged = [float(pose_7d_m[0]), float(pose_7d_m[1]), float(pose_7d_m[2]), *current[3:7]]
        print("Using current left TCP orientation for target pose:", merged)
        return merged

    def move_left_to_pose(
        self,
        pose_7d_m: list[float],
        *,
        vel: float,
        acc: float,
        planner: str,
    ) -> None:
        if len(pose_7d_m) != 7:
            raise ValueError(f"Expected 7D target pose, got {len(pose_7d_m)} values: {pose_7d_m}")
        result = self._robot.motion.move_l(
            arm=self._arm_type.Left,
            pose=[float(value) for value in pose_7d_m],
            vel=vel,
            acc=acc,
            wait=True,
            planner=planner,
        )
        print("Grasp move_l result:", result.success, result.message)
        if not result:
            raise RuntimeError(f"YSRobot move_l failed: {result.message}")

    def close_left_gripper(self, *, register: int, close_value: int, settle_s: float) -> None:
        result = self._robot.device.write_modbus(int(register), int(close_value))
        print("Gripper close result:", result.success, result.message)
        if not result:
            raise RuntimeError(f"gripper CLOSE failed: {result.message}")
        time.sleep(float(settle_s))

    def close(self) -> None:
        self._robot.disconnect()


class MockLeftTcpPoseClient:
    def get_left_xyz(self) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0

    def get_left_pose_7d(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def with_current_left_orientation(self, pose_7d_m: list[float]) -> list[float]:
        return [float(pose_7d_m[0]), float(pose_7d_m[1]), float(pose_7d_m[2]), 0.0, 0.0, 0.0, 1.0]

    def close(self) -> None:
        pass

    def move_left_to_pose(
        self,
        pose_7d_m: list[float],
        *,
        vel: float,
        acc: float,
        planner: str,
    ) -> None:
        print(f"[mock] move_l left pose={pose_7d_m}, vel={vel}, acc={acc}, planner={planner}")

    def close_left_gripper(self, *, register: int, close_value: int, settle_s: float) -> None:
        print(f"[mock] close gripper register={register}, value={close_value}, settle_s={settle_s}")


class InterruptionMonitor:
    """Requests VLA pauses when bottle and left TCP are close enough."""

    def __init__(
        self,
        config: InterruptionConfig,
        *,
        yolo: YoloHttpClient,
        pose_client: LeftTcpPoseClient | MockLeftTcpPoseClient,
        runtime_control: RuntimeControl,
    ) -> None:
        self._config = config
        self._yolo = yolo
        self._pose_client = pose_client
        self._runtime_control = runtime_control
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._interruptions = 0
        self._last_interrupt_time = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="vla_interruption_monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        period_s = 1.0 / max(float(self._config.poll_hz), 1e-6)
        while not self._stop.is_set() and not self._runtime_control.stop_event.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                print(f"Interruption monitor error: {exc}")
            time.sleep(period_s)

    def _poll_once(self) -> None:
        if self._runtime_control.pause_event.is_set() or self._runtime_control.paused_event.is_set():
            return
        if self._config.max_interruptions > 0 and self._interruptions >= self._config.max_interruptions:
            return
        now = time.monotonic()
        if now - self._last_interrupt_time < float(self._config.cooldown_s):
            return

        bottle_xyz = self._yolo.get_bottle_xyz()
        if bottle_xyz is None:
            return
        left_xyz = self._pose_client.get_left_xyz()
        distance = _distance(left_xyz, bottle_xyz)
        print(
            "VLA interruption monitor: "
            f"left_tcp={_fmt_xyz(left_xyz)}, bottle={_fmt_xyz(bottle_xyz)}, distance={distance:.4f}m"
        )
        if distance <= float(self._config.distance_threshold_m):
            self._interruptions += 1
            self._last_interrupt_time = now
            print(
                "VLA interruption trigger: "
                f"distance={distance:.4f}m <= threshold={self._config.distance_threshold_m:.4f}m"
            )
            self._runtime_control.request_pause()


def _grasp_policy(
    runtime_control: RuntimeControl,
    *,
    yolo: YoloHttpClient,
    pose_client: LeftTcpPoseClient | MockLeftTcpPoseClient,
    ros_io: NZ100Ros2IO | None,
    config: InterruptionConfig,
) -> None:
    """Middle grasp slot: move left TCP to the latest vision target pose."""

    print("VLA paused. Requesting latest bottle grasp target for middle policy...")
    target = yolo.get_bottle_target()
    if target is None:
        print("No valid bottle grasp target; skipping grasp policy and resuming VLA.")
        runtime_control.resume()
        return
    print("Bottle grasp target pose_7d_m:", target.pose_7d_m)
    grasp_pose = pose_client.with_current_left_orientation(target.pose_7d_m)
    if ros_io is not None and bool(config.handoff_hold_enabled):
        ros_io.hold_current_joint_positions(duration_s=float(config.handoff_hold_s))
    print("\n[1/3] move_l -> GRASP")
    pose_client.move_left_to_pose(
        grasp_pose,
        vel=float(config.grasp_move_vel),
        acc=float(config.grasp_move_acc),
        planner=str(config.grasp_move_planner),
    )
    print("\n[2/3] CLOSE gripper")
    pose_client.close_left_gripper(
        register=int(config.gripper_register),
        close_value=int(config.gripper_close_value),
        settle_s=float(config.gripper_settle_s),
    )
    if ros_io is not None:
        ros_io.set_cached_gripper_state(left=float(config.gripper_close_value))
    if not bool(config.no_lift):
        print("\n[3/3] move_l -> LIFT")
        lift_target = VisionTarget(grasp_pose)
        lift_pose = yolo.get_lift_pose(lift_target, lift_mm=float(config.lift_mm))
        pose_client.move_left_to_pose(
            lift_pose,
            vel=float(config.grasp_move_vel),
            acc=float(config.grasp_move_acc),
            planner=str(config.grasp_move_planner),
        )
    else:
        print("\n[3/3] LIFT skipped by no_lift=True")
    print("Middle grasp policy finished; resuming VLA.")
    runtime_control.resume()


def _extract_xyz(payload: Any) -> tuple[float, float, float] | None:
    """Accept common YOLO service response shapes."""

    if isinstance(payload, dict):
        for key in ("xyz", "center_xyz", "bottle_xyz", "center", "position"):
            xyz = _extract_xyz(payload.get(key))
            if xyz is not None:
                return xyz
        if all(key in payload for key in ("x", "y", "z")):
            return float(payload["x"]), float(payload["y"]), float(payload["z"])
        data = payload.get("data")
        if data is not payload:
            return _extract_xyz(data)
    if isinstance(payload, (list, tuple)) and len(payload) >= 3:
        return float(payload[0]), float(payload[1]), float(payload[2])
    return None


def _extract_pose_7d(payload: Any) -> list[float] | None:
    """Parse the bottle service response and return target.pose_7d_m."""

    if not isinstance(payload, dict):
        xyz = _extract_xyz(payload)
        if xyz is None:
            return None
        return [*xyz, 0.0, 0.0, 0.0, 1.0]

    if "valid" in payload and not bool(payload.get("valid")):
        print(f"Bottle vision target invalid: {payload.get('reason')}")
        return None

    target = payload.get("target")
    if isinstance(target, dict) and isinstance(target.get("pose_7d_m"), (list, tuple)):
        pose = [float(value) for value in target["pose_7d_m"]]
        if len(pose) >= 7:
            return pose[:7]
        print(f"Bottle vision pose_7d_m has too few values: {pose}")
        return None

    for key in ("pose_7d_m", "target_pose", "pose"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 7:
            return [float(v) for v in value[:7]]

    xyz = _extract_xyz(payload)
    if xyz is not None:
        return [*xyz, 0.0, 0.0, 0.0, 1.0]
    return None


def _http_get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(payload).__name__}")
    return payload


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _fmt_xyz(xyz: tuple[float, float, float]) -> str:
    return f"({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})"


def _build_client_config(client_base: ClientConfig) -> ClientConfig:
    supported_modes = {"sync_chunk", "async_queue", "rtc_guidance", "legato"}
    if client_base.execution_mode not in supported_modes:
        raise ValueError(
            "multi_stage_control interruption framework currently supports "
            f"execution_mode={sorted(supported_modes)!r}, "
            f"got {client_base.execution_mode!r}"
        )
    return dataclasses.replace(client_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuous VLA with perception-triggered interruption")
    parser.add_argument("config", type=Path, nargs="?", default=Path(__file__).with_name("stages.yaml"))
    parser.add_argument("--mock", action="store_true", help="Use mock robot observation in the VLA runner")
    args = parser.parse_args()

    interruption_config = load_interruption_config(args.config)
    app_config = load_app_config(interruption_config.robot_client_config)
    runtime_control = RuntimeControl()

    ros_io = None
    pose_client: LeftTcpPoseClient | MockLeftTcpPoseClient | None = None
    monitor = None
    try:
        if not args.mock:
            ros_io = NZ100Ros2IO(app_config.ros2)
            ros_io.connect()
            if app_config.ros2.home_on_start:
                ros_io.move_to_home()
            else:
                print("Skipping NZ100 startup pose command.")

        if args.mock:
            pose_client = MockLeftTcpPoseClient()
        else:
            pose_client = LeftTcpPoseClient(app_config, frame=interruption_config.left_pose_frame)
        yolo = YoloHttpClient(
            interruption_config.yolo_url,
            timeout_s=float(interruption_config.yolo_timeout_s),
            mock_xyz=interruption_config.mock_yolo_xyz,
        )
        monitor = InterruptionMonitor(
            interruption_config,
            yolo=yolo,
            pose_client=pose_client,
            runtime_control=runtime_control,
        )
        runtime_control.on_pause = lambda: _grasp_policy(
            runtime_control,
            yolo=yolo,
            pose_client=pose_client,
            ros_io=ros_io,
            config=interruption_config,
        )
        monitor.start()

        print("Starting continuous VLA with interruption monitor.")
        client_config = _build_client_config(app_config.client)
        if client_config.execution_mode == "sync_chunk":
            sync_chunk.run(
                client_config,
                ros_io=ros_io,
                mock=args.mock,
                once=False,
                runtime_control=runtime_control,
            )
        elif client_config.execution_mode == "async_queue":
            async_queue.run(
                client_config,
                ros_io=ros_io,
                mock=args.mock,
                once=False,
                runtime_control=runtime_control,
            )
        elif client_config.execution_mode == "rtc_guidance":
            rtc_guidance.run(
                client_config,
                ros_io=ros_io,
                mock=args.mock,
                once=False,
                runtime_control=runtime_control,
            )
        elif client_config.execution_mode == "legato":
            legato.run(
                client_config,
                ros_io=ros_io,
                mock=args.mock,
                once=False,
                runtime_control=runtime_control,
            )
        else:
            raise ValueError(f"Unsupported execution_mode: {client_config.execution_mode!r}")
    finally:
        runtime_control.request_stop()
        if monitor is not None:
            monitor.stop()
        if pose_client is not None:
            pose_client.close()
        if ros_io is not None:
            ros_io.disconnect()


if __name__ == "__main__":
    main()
