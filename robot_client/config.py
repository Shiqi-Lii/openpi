"""Configuration for the NZ100 robot-side policy client."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class ClientConfig:
    """Network and runtime settings for connecting to the OpenPI policy server."""

    server_host: str = "172.22.1.127"
    server_port: int = 5555
    prompt: str = "pick up the bottle and place it in the blue box"
    image_size: int = 224
    control_hz: float = 100.0
    open_loop_horizon: int = 32
    max_steps: int = 0
    execution_mode: str = "sync_chunk"

    # For normal chunk execution, the client requests one action chunk, then
    # executes actions from that chunk locally before asking the server again.
    execute_full_chunk: bool = True

    # Real-Time Chunking settings. These are only used when
    # ``execution_mode`` is ``rtc_prefix`` or ``rtc_guidance``.
    rtc_execute_horizon: int = 8
    rtc_prefix_len: int = 5
    rtc_guidance_weight: float = 5.0
    rtc_decay_tau: float = 3.0


@dataclasses.dataclass(frozen=True)
class Ros2Config:
    """ROS 2 topics and robot IO settings for NZ100."""

    top_camera_topic: str = "/top/image_raw"
    wrist_left_camera_topic: str = "/wrist_left/image_raw"
    joint_state_topic: str = "/joint_states"
    left_trajectory_topic: str = "/arm_left_controller/joint_trajectory"
    right_trajectory_topic: str = "/arm_right_controller/joint_trajectory"

    left_joint_names: tuple[str, ...] = (
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "left_joint6",
        "left_joint7",
    )
    right_joint_names: tuple[str, ...] = (
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_joint6",
        "right_joint7",
    )

    point_time_from_start: float = 0.01

    # Optional startup homing. When enabled, both arm trajectories are
    # published together and policy inference starts after both are complete.
    home_on_start: bool = True
    left_home_positions: tuple[float, ...] = (0.36, 0.36, -0.01, 1.92, 1.57, 0.0, -1.40)
    right_home_positions: tuple[float, ...] = (-0.36, 0.36, -0.01, 1.92, 1.57, 0.0, 0.78)
    home_time_from_start: float = 4.0

    # Dual-gripper Modbus topics. This matches the data collection setup.
    gripper_state_topic: str = "/robot/api/io/state"
    gripper_cmd_topic: str = "/robot/api/io/cmd"
    left_gripper_key: str = "an_out_d9746"
    right_gripper_key: str = "an_out_d9747"

    # Data collection stores PLC gripper semantics as 1=open, 2=closed.
    modbus_open_value: int = 1
    modbus_closed_value: int = 2
    gripper_default_value: float = 1.0


@dataclasses.dataclass(frozen=True)
class AppConfig:
    client: ClientConfig = dataclasses.field(default_factory=ClientConfig)
    ros2: Ros2Config = dataclasses.field(default_factory=Ros2Config)


def load_app_config(path: str | Path | None) -> AppConfig:
    """Load YAML config. Missing keys keep dataclass defaults.

    Supports both:
    1. the current flat robot-client config style, e.g. ``policy_host``;
    2. the older nested style, e.g. ``client.server_host``.
    """

    if path is None:
        return AppConfig()

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for --config. Install with: pip install pyyaml") from exc

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")

    client_data = {
        **_flat_client_data(data),
        **_as_mapping(data.get("client", {})),
    }
    ros2_data = {
        **_flat_ros2_data(data),
        **_as_mapping(data.get("ros2", {})),
    }
    return AppConfig(
        client=ClientConfig(**_filter_dataclass_kwargs(ClientConfig, client_data)),
        ros2=Ros2Config(**_filter_dataclass_kwargs(Ros2Config, ros2_data)),
    )


def _flat_client_data(data: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "policy_host": "server_host",
        "policy_port": "server_port",
        "control_fps": "control_hz",
        "open_loop_horizon": "open_loop_horizon",
        "max_steps": "max_steps",
        "language_instruction": "prompt",
        "execution_mode": "execution_mode",
        "rtc_execute_horizon": "rtc_execute_horizon",
        "rtc_prefix_len": "rtc_prefix_len",
        "rtc_guidance_weight": "rtc_guidance_weight",
        "rtc_decay_tau": "rtc_decay_tau",
    }
    result = {target: data[source] for source, target in mapping.items() if source in data}
    if result.get("execution_mode") == "async_queue":
        # Backward-compatible alias for the normal synchronous chunk mode.
        result["execution_mode"] = "sync_chunk"
    return result


def _flat_ros2_data(data: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "point_time_from_start",
        "top_camera_topic",
        "wrist_left_camera_topic",
        "joint_state_topic",
        "left_trajectory_topic",
        "right_trajectory_topic",
        "left_joint_names",
        "right_joint_names",
        "home_on_start",
        "left_home_positions",
        "right_home_positions",
        "home_time_from_start",
        "gripper_state_topic",
        "gripper_cmd_topic",
        "left_gripper_key",
        "right_gripper_key",
        "modbus_open_value",
        "modbus_closed_value",
        "gripper_default_value",
    }
    return {key: data[key] for key in keys if key in data}


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping, got {type(value).__name__}")
    return value


def _filter_dataclass_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    valid = {field.name for field in dataclasses.fields(cls)}
    result = {key: _maybe_tuple(value) for key, value in data.items() if key in valid}
    if result.get("execution_mode") == "async_queue":
        result["execution_mode"] = "sync_chunk"
    return result


def _maybe_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value
