"""Entry point for normal synchronous NZ100 policy inference.

The mock mode is safe for checking network/model connectivity.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
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


@dataclasses.dataclass
class RTCActionContext:
    """Timing and action chunk state needed for real-time chunking."""

    raw_chunk: np.ndarray
    inference_start_s: float
    inference_elapsed_s: float
    action_timestamp_s: float


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
        rtc_decay_end=client_base.rtc_decay_end,
        rtc_use_vjp=client_base.rtc_use_vjp,
    )

    print(
        "Starting NZ100 OpenPI client: "
        f"server=tcp://{config.server_host}:{config.server_port}, "
        f"mode={config.execution_mode}, "
        f"control_hz={config.control_hz}, "
        f"open_loop_horizon={config.open_loop_horizon}, "
        f"max_steps={config.max_steps}, "
        f"mock={args.mock}"
    )
    print(f"Language instruction: {config.prompt!r}")

    ros_io = None
    try:
        if not args.mock:
            ros_io = NZ100Ros2IO(app_config.ros2)
            ros_io.connect()
            if app_config.ros2.home_on_start:
                ros_io.move_to_home()
            else:
                print("Skipping NZ100 startup pose command.")

        if config.execution_mode == "sync_chunk":
            _run_sync_loop(config, ros_io=ros_io, mock=args.mock, once=args.once)
        elif config.execution_mode in ("rtc_prefix", "rtc_guidance"):
            _run_rtc_loop(config, ros_io=ros_io, mock=args.mock, once=args.once)
        else:
            raise ValueError(f"Unsupported execution_mode: {config.execution_mode!r}")
    finally:
        if ros_io is not None:
            ros_io.disconnect()


def _read_observation(
    ros_io: NZ100Ros2IO | None, *, mock: bool
) -> tuple[np.ndarray, np.ndarray, NZ100RobotState]:
    top_image = read_mock_top_image() if mock else ros_io.get_top_image()
    wrist_left_image = read_mock_wrist_left_image() if mock else ros_io.get_wrist_left_image()
    robot_state = read_mock_robot_state() if mock else ros_io.get_robot_state()
    return top_image, wrist_left_image, robot_state


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
        print(f"Executing action[{executed_steps}]: {_format_action(action)}")
        if mock:
            print(action)
        else:
            ros_io.apply_action(action)

        executed_steps += 1
        if step_sleep > 0:
            time.sleep(step_sleep)
    return executed_steps


def _resample_remaining(sequence: np.ndarray, offset: float) -> np.ndarray:
    """Linearly interpolate the remaining chunk from a fractional timestep offset."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected action chunk with shape (T, D), got {sequence.shape}")

    offset = max(float(offset), 0.0)
    length = sequence.shape[0]
    remaining_len = length - int(offset)
    if remaining_len <= 0:
        return sequence[:0]

    indices = np.clip(offset + np.arange(remaining_len, dtype=np.float32), 0.0, float(length - 1))
    lo = np.floor(indices).astype(np.int64)
    hi = np.minimum(lo + 1, length - 1)
    alpha = (indices - lo)[:, None]
    return sequence[lo] + alpha * (sequence[hi] - sequence[lo])


def _steps_from_elapsed(elapsed_s: float, control_hz: float) -> int:
    if control_hz <= 0:
        return 0
    return max(0, int(elapsed_s * control_hz))


def _compute_rtc_prefix_len(config: ClientConfig, inference_elapsed_s: float, remaining_len: int) -> int:
    if remaining_len <= 0:
        return 0
    configured = int(config.rtc_prefix_len)
    latency_steps = _steps_from_elapsed(inference_elapsed_s, config.control_hz)
    prefix_len = max(configured, latency_steps)
    return min(prefix_len, remaining_len)


def _remaining_from_context(ctx: RTCActionContext, *, now_s: float, control_hz: float) -> np.ndarray:
    offset = (now_s - ctx.action_timestamp_s) * control_hz if control_hz > 0 else 0.0
    return _resample_remaining(ctx.raw_chunk, offset)


def _run_sync_loop(config: ClientConfig, *, ros_io: NZ100Ros2IO | None, mock: bool, once: bool) -> None:
    client = NZ100SyncClient(config)
    executed_steps = 0
    print("Entering sync_chunk control loop.")

    while True:
        print(f"Reading observation before request; executed_steps={executed_steps}")
        top_image, wrist_left_image, robot_state = _read_observation(ros_io, mock=mock)
        print(f"Requesting action chunk from OpenPI server; state={_format_state(robot_state)}")
        tic = time.time()
        action_chunk = client.infer(
            top_image=top_image, wrist_left_image=wrist_left_image, robot_state=robot_state
        )
        print(
            "Received action chunk: "
            f"shape={tuple(action_chunk.shape)}, latency={time.time() - tic:.3f}s"
        )
        if config.open_loop_horizon > 0:
            action_chunk = action_chunk[: config.open_loop_horizon]
        if not config.execute_full_chunk:
            action_chunk = action_chunk[:1]
        print(f"Executing {len(action_chunk)} actions from current chunk.")

        executed_steps = _execute_action_chunk(
            action_chunk,
            config=config,
            ros_io=ros_io,
            mock=mock,
            executed_steps=executed_steps,
        )
        if config.max_steps > 0 and executed_steps >= config.max_steps:
            print(f"Reached max_steps={config.max_steps}; stopping.")
            return
        if once:
            print("--once enabled; stopping after one chunk.")
            return


def _run_rtc_loop(config: ClientConfig, *, ros_io: NZ100Ros2IO | None, mock: bool, once: bool) -> None:
    client = NZ100RTCClient(config)
    executed_steps = 0
    print(f"Entering {config.execution_mode} control loop.")

    print("Reading initial observation for RTC.")
    top_image, wrist_left_image, robot_state = _read_observation(ros_io, mock=mock)
    print(f"Requesting initial RTC action chunk; state={_format_state(robot_state)}")
    inference_start_s = time.time()
    current_chunk = client.infer(
        top_image=top_image, wrist_left_image=wrist_left_image, robot_state=robot_state
    )
    inference_elapsed_s = time.time() - inference_start_s
    print(
        "Received initial RTC chunk: "
        f"shape={tuple(current_chunk.shape)}, latency={inference_elapsed_s:.3f}s"
    )
    if config.open_loop_horizon > 0:
        current_chunk = current_chunk[: config.open_loop_horizon]
    ctx = RTCActionContext(
        raw_chunk=current_chunk,
        inference_start_s=inference_start_s,
        inference_elapsed_s=inference_elapsed_s,
        action_timestamp_s=time.time(),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            current_chunk = _remaining_from_context(ctx, now_s=time.time(), control_hz=config.control_hz)
            execute_horizon = min(int(config.rtc_execute_horizon), current_chunk.shape[0])
            if execute_horizon <= 0:
                print("Current RTC chunk is exhausted; requesting a fresh chunk.")
                top_image, wrist_left_image, robot_state = _read_observation(ros_io, mock=mock)
                inference_start_s = time.time()
                current_chunk = client.infer(
                    top_image=top_image,
                    wrist_left_image=wrist_left_image,
                    robot_state=robot_state,
                )
                inference_elapsed_s = time.time() - inference_start_s
                if config.open_loop_horizon > 0:
                    current_chunk = current_chunk[: config.open_loop_horizon]
                ctx = RTCActionContext(
                    raw_chunk=current_chunk,
                    inference_start_s=inference_start_s,
                    inference_elapsed_s=inference_elapsed_s,
                    action_timestamp_s=time.time(),
                )
                continue

            next_top_image, next_wrist_left_image, next_robot_state = _read_observation(
                ros_io, mock=mock
            )
            next_inference_start_s = time.time()
            remaining_for_guidance = _remaining_from_context(
                ctx, now_s=next_inference_start_s, control_hz=config.control_hz
            )
            prefix_len = _compute_rtc_prefix_len(
                config, ctx.inference_elapsed_s, remaining_for_guidance.shape[0]
            )
            print(
                "Requesting next RTC chunk in background: "
                f"executed_steps={executed_steps}, "
                f"remaining={remaining_for_guidance.shape[0]}, "
                f"prefix_len={prefix_len}, "
                f"state={_format_state(next_robot_state)}"
            )
            next_future = executor.submit(
                client.infer,
                top_image=next_top_image,
                wrist_left_image=next_wrist_left_image,
                robot_state=next_robot_state,
                previous_chunk=remaining_for_guidance,
                prefix_len=prefix_len,
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
                print(f"Reached max_steps={config.max_steps}; stopping.")
                return
            if once:
                print("--once enabled; stopping after one RTC execution window.")
                return

            current_chunk = next_future.result()
            inference_elapsed_s = time.time() - next_inference_start_s
            print(
                "Received next RTC chunk: "
                f"shape={tuple(current_chunk.shape)}, latency={inference_elapsed_s:.3f}s"
            )
            if config.open_loop_horizon > 0:
                current_chunk = current_chunk[: config.open_loop_horizon]
            skip_steps = _steps_from_elapsed(inference_elapsed_s, config.control_hz)
            if skip_steps > 0:
                print(f"Skipping {skip_steps} expired RTC actions from new chunk.")
            current_chunk = current_chunk[skip_steps:]
            ctx = RTCActionContext(
                raw_chunk=current_chunk,
                inference_start_s=next_inference_start_s,
                inference_elapsed_s=inference_elapsed_s,
                action_timestamp_s=time.time(),
            )


def _format_state(state: NZ100RobotState) -> str:
    return (
        f"left={_format_array(state.left_joints)}, "
        f"left_gripper={state.left_gripper:.1f}, "
        f"right={_format_array(state.right_joints)}, "
        f"right_gripper={state.right_gripper:.1f}"
    )


def _format_action(action) -> str:
    return (
        f"left={_format_array(action.left_joints)}, "
        f"left_gripper={action.left_gripper:.1f}, "
        f"right={_format_array(action.right_joints)}, "
        f"right_gripper={action.right_gripper:.1f}"
    )


def _format_array(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values, dtype=np.float32), precision=3, suppress_small=True)


if __name__ == "__main__":
    main()
