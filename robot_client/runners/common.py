"""Shared helpers for NZ100 robot-client runners."""

from __future__ import annotations

import time

import numpy as np

from robot_client.config import ClientConfig
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.state_builder import NZ100RobotState
from robot_client.state_builder import discretize_plc_grippers
from robot_client.state_builder import split_action
from robot_client.sync_client import NZ100SyncClient


def read_mock_top_image() -> np.ndarray:
    return np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)


def read_mock_wrist_left_image() -> np.ndarray:
    return np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)


def read_mock_robot_state() -> NZ100RobotState:
    return NZ100RobotState(
        left_joints=np.zeros((7,), dtype=np.float32),
        right_joints=np.zeros((7,), dtype=np.float32),
        left_gripper=1.0,
        right_gripper=1.0,
    )


def read_observation(
    ros_io: NZ100Ros2IO | None, *, mock: bool
) -> tuple[np.ndarray, np.ndarray, NZ100RobotState]:
    top_image = read_mock_top_image() if mock else ros_io.get_top_image()
    wrist_left_image = read_mock_wrist_left_image() if mock else ros_io.get_wrist_left_image()
    robot_state = read_mock_robot_state() if mock else ros_io.get_robot_state()
    return top_image, wrist_left_image, robot_state


def execute_action_chunk(
    action_chunk: np.ndarray,
    *,
    config: ClientConfig,
    ros_io: NZ100Ros2IO | None,
    mock: bool,
    executed_steps: int,
) -> int:
    step_sleep = 1.0 / config.control_hz if config.control_hz > 0 else 0.0
    for raw_action in action_chunk:
        action = discretize_plc_grippers(split_action(raw_action))
        print(f"Executing action[{executed_steps}]: {format_action(action)}")
        if mock:
            print(action)
        else:
            ros_io.apply_action(action)

        executed_steps += 1
        if step_sleep > 0:
            time.sleep(step_sleep)
    return executed_steps


def infer_sync_chunk(
    client: NZ100SyncClient,
    config: ClientConfig,
    ros_io: NZ100Ros2IO | None,
    *,
    mock: bool,
    log_prefix: str = "",
) -> np.ndarray:
    top_image, wrist_left_image, robot_state = read_observation(ros_io, mock=mock)
    print(f"{log_prefix}Requesting action chunk from OpenPI server; state={format_state(robot_state)}")
    tic = time.monotonic()
    action_chunk = client.infer(
        top_image=top_image,
        wrist_left_image=wrist_left_image,
        robot_state=robot_state,
    )
    print(
        f"{log_prefix}Received action chunk: "
        f"shape={tuple(action_chunk.shape)}, latency={time.monotonic() - tic:.3f}s"
    )
    if config.open_loop_horizon > 0:
        action_chunk = action_chunk[: config.open_loop_horizon]
    return action_chunk


def format_state(state: NZ100RobotState) -> str:
    return (
        f"left={format_array(state.left_joints)}, "
        f"left_gripper={state.left_gripper:.1f}, "
        f"right={format_array(state.right_joints)}, "
        f"right_gripper={state.right_gripper:.1f}"
    )


def format_action(action) -> str:
    return (
        f"left={format_array(action.left_joints)}, "
        f"left_gripper={action.left_gripper:.1f}, "
        f"right={format_array(action.right_joints)}, "
        f"right_gripper={action.right_gripper:.1f}"
    )


def format_array(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values, dtype=np.float32), precision=3, suppress_small=True)

