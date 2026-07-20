"""Entry point for normal synchronous NZ100 policy inference.

The mock mode is safe for checking network/model connectivity.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import time
from pathlib import Path

import numpy as np

from robot_client.config import ClientConfig
from robot_client.config import load_app_config
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.state_builder import NZ100RobotState
from robot_client.state_builder import discretize_plc_grippers
from robot_client.state_builder import split_action
from robot_client.rtc_client import NZ100RTCClient
from robot_client.sync_client import NZ100SyncClient


def read_mock_top_image() -> np.ndarray:
    return np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)


def read_mock_robot_state() -> NZ100RobotState:
    return NZ100RobotState(
        left_joints=np.zeros((7,), dtype=np.float32),
        right_joints=np.zeros((7,), dtype=np.float32),
        left_gripper=1.0,
        right_gripper=1.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NZ100 normal synchronous OpenPI client")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("robot_client/configs/nz100_client.yaml"),
        help="YAML config for server and ROS2 topics",
    )
    parser.add_argument("--host", default=None, help="Override GPU policy server IP/hostname")
    parser.add_argument("--port", type=int, default=None, help="Override GPU policy server port")
    parser.add_argument("--prompt", default=None, help="Override language instruction")
    parser.add_argument("--control-hz", type=float, default=None, help="Override local action execution rate")
    parser.add_argument(
        "--execution-mode",
        choices=("sync_chunk", "rtc_prefix", "rtc_guidance"),
        default=None,
        help="Override inference/execution mode",
    )
    parser.add_argument("--mock", action="store_true", help="Use fake camera/state and print actions only")
    parser.add_argument("--once", action="store_true", help="Run one inference request and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = load_app_config(args.config)

    client_base = app_config.client
    config = ClientConfig(
        server_host=client_base.server_host if args.host is None else args.host,
        server_port=client_base.server_port if args.port is None else args.port,
        prompt=client_base.prompt if args.prompt is None else args.prompt,
        image_size=client_base.image_size,
        control_hz=client_base.control_hz if args.control_hz is None else args.control_hz,
        open_loop_horizon=client_base.open_loop_horizon,
        max_steps=client_base.max_steps,
        execution_mode=client_base.execution_mode if args.execution_mode is None else args.execution_mode,
        execute_full_chunk=client_base.execute_full_chunk,
        rtc_execute_horizon=client_base.rtc_execute_horizon,
        rtc_prefix_len=client_base.rtc_prefix_len,
        rtc_guidance_weight=client_base.rtc_guidance_weight,
        rtc_decay_tau=client_base.rtc_decay_tau,
    )

    ros_io = None
    try:
        if not args.mock:
            ros_io = NZ100Ros2IO(app_config.ros2)
            ros_io.connect()
            if app_config.ros2.home_on_start:
                ros_io.move_to_home()

        if config.execution_mode == "sync_chunk":
            _run_sync_loop(config, ros_io=ros_io, mock=args.mock, once=args.once)
        elif config.execution_mode in ("rtc_prefix", "rtc_guidance"):
            _run_rtc_loop(config, ros_io=ros_io, mock=args.mock, once=args.once)
        else:
            raise ValueError(f"Unsupported execution_mode: {config.execution_mode!r}")
    finally:
        if ros_io is not None:
            ros_io.disconnect()


def _read_observation(ros_io: NZ100Ros2IO | None, *, mock: bool) -> tuple[np.ndarray, NZ100RobotState]:
    top_image = read_mock_top_image() if mock else ros_io.get_top_image()
    robot_state = read_mock_robot_state() if mock else ros_io.get_robot_state()
    return top_image, robot_state


def _execute_action_chunk(
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
        if mock:
            print(action)
        else:
            ros_io.apply_action(action)

        executed_steps += 1
        if step_sleep > 0:
            time.sleep(step_sleep)
    return executed_steps


def _run_sync_loop(config: ClientConfig, *, ros_io: NZ100Ros2IO | None, mock: bool, once: bool) -> None:
    client = NZ100SyncClient(config)
    executed_steps = 0

    while True:
        top_image, robot_state = _read_observation(ros_io, mock=mock)
        action_chunk = client.infer(top_image=top_image, robot_state=robot_state)
        if config.open_loop_horizon > 0:
            action_chunk = action_chunk[: config.open_loop_horizon]
        if not config.execute_full_chunk:
            action_chunk = action_chunk[:1]

        executed_steps = _execute_action_chunk(
            action_chunk,
            config=config,
            ros_io=ros_io,
            mock=mock,
            executed_steps=executed_steps,
        )
        if config.max_steps > 0 and executed_steps >= config.max_steps:
            return
        if once:
            return


def _run_rtc_loop(config: ClientConfig, *, ros_io: NZ100Ros2IO | None, mock: bool, once: bool) -> None:
    client = NZ100RTCClient(config)
    executed_steps = 0

    top_image, robot_state = _read_observation(ros_io, mock=mock)
    current_chunk = client.infer(top_image=top_image, robot_state=robot_state)
    if config.open_loop_horizon > 0:
        current_chunk = current_chunk[: config.open_loop_horizon]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            execute_horizon = min(int(config.rtc_execute_horizon), current_chunk.shape[0])
            if execute_horizon <= 0:
                raise ValueError("rtc_execute_horizon must be positive")

            next_top_image, next_robot_state = _read_observation(ros_io, mock=mock)
            next_future = executor.submit(
                client.infer,
                top_image=next_top_image,
                robot_state=next_robot_state,
                previous_chunk=current_chunk,
            )

            to_execute = current_chunk[:execute_horizon]
            executed_steps = _execute_action_chunk(
                to_execute,
                config=config,
                ros_io=ros_io,
                mock=mock,
                executed_steps=executed_steps,
            )
            if config.max_steps > 0 and executed_steps >= config.max_steps:
                return
            if once:
                return

            current_chunk = next_future.result()
            if config.open_loop_horizon > 0:
                current_chunk = current_chunk[: config.open_loop_horizon]


if __name__ == "__main__":
    main()
