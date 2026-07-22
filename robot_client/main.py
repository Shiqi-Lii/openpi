"""Entry point for normal synchronous NZ100 policy inference.

The mock mode is safe for checking network/model connectivity.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import threading
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
    step_index: int = 0


@dataclasses.dataclass
class RTCSharedState:
    """Mutex-protected RTC state shared by controller and inference threads."""

    ctx: RTCActionContext
    stop: bool = False
    error: BaseException | None = None


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
        rtc_delay_buffer_size=client_base.rtc_delay_buffer_size,
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


def _steps_from_elapsed(elapsed_s: float, control_hz: float) -> int:
    if control_hz <= 0:
        return 0
    return max(0, int(elapsed_s * control_hz))


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
    if config.control_hz <= 0:
        raise ValueError(f"control_hz must be positive for RTC, got {config.control_hz}")

    print("Reading initial observation for RTC.")
    top_image, wrist_left_image, robot_state = _read_observation(ros_io, mock=mock)
    print(f"Requesting initial RTC action chunk; state={_format_state(robot_state)}")
    inference_start_s = time.monotonic()
    current_chunk = client.infer(
        top_image=top_image, wrist_left_image=wrist_left_image, robot_state=robot_state
    )
    inference_elapsed_s = time.monotonic() - inference_start_s
    print(
        "Received initial RTC chunk: "
        f"shape={tuple(current_chunk.shape)}, latency={inference_elapsed_s:.3f}s"
    )
    if config.open_loop_horizon > 0:
        current_chunk = current_chunk[: config.open_loop_horizon]
    if current_chunk.shape[0] == 0:
        raise ValueError("Initial RTC action chunk is empty")

    condition = threading.Condition()
    shared = RTCSharedState(ctx=RTCActionContext(raw_chunk=current_chunk, step_index=0))
    delay_buffer = collections.deque(
        [_steps_from_elapsed(inference_elapsed_s, config.control_hz)],
        maxlen=max(1, int(config.rtc_delay_buffer_size)),
    )
    inference_thread = threading.Thread(
        target=_rtc_inference_loop,
        args=(config, client, ros_io, mock, shared, condition, delay_buffer),
        daemon=True,
    )
    inference_thread.start()

    period_s = 1.0 / config.control_hz
    next_tick = time.monotonic()
    try:
        while True:
            with condition:
                if shared.error is not None:
                    raise RuntimeError("RTC background inference failed") from shared.error
                ctx = shared.ctx
                if ctx.step_index < ctx.raw_chunk.shape[0]:
                    raw_action = np.asarray(ctx.raw_chunk[ctx.step_index], dtype=np.float32)
                    ctx.step_index += 1
                    local_step_index = ctx.step_index
                    condition.notify_all()
                else:
                    raw_action = np.asarray(ctx.raw_chunk[-1], dtype=np.float32)
                    local_step_index = ctx.step_index
                    condition.notify_all()
                    print("RTC chunk exhausted before replacement; holding last action.")

            action = discretize_plc_grippers(split_action(raw_action))
            print(
                f"Executing RTC action[{executed_steps}] "
                f"chunk_step={local_step_index}: {_format_action(action)}"
            )
            if mock:
                print(action)
            else:
                ros_io.apply_action(action)

            executed_steps += 1
            if config.max_steps > 0 and executed_steps >= config.max_steps:
                print(f"Reached max_steps={config.max_steps}; stopping.")
                return
            if once:
                print("--once enabled; stopping after one RTC action.")
                return

            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    finally:
        with condition:
            shared.stop = True
            condition.notify_all()
        inference_thread.join(timeout=2.0)


def _rtc_inference_loop(
    config: ClientConfig,
    client: NZ100RTCClient,
    ros_io: NZ100Ros2IO | None,
    mock: bool,
    shared: RTCSharedState,
    condition: threading.Condition,
    delay_buffer: collections.deque[int],
) -> None:
    """Background inference loop following RTC Algorithm 1's t/s/d structure."""

    minimum_execution_horizon = max(1, int(config.rtc_execute_horizon))
    while True:
        with condition:
            condition.wait_for(
                lambda: shared.stop
                or shared.error is not None
                or shared.ctx.step_index >= minimum_execution_horizon
                or shared.ctx.step_index >= shared.ctx.raw_chunk.shape[0]
            )
            if shared.error is not None:
                return
            if shared.stop:
                return

            start_step = int(shared.ctx.step_index)
            previous_chunk = np.asarray(shared.ctx.raw_chunk[start_step:], dtype=np.float32).copy()
            predicted_delay_steps = max(max(delay_buffer), int(config.rtc_prefix_len))

        try:
            top_image, wrist_left_image, robot_state = _read_observation(ros_io, mock=mock)
            prefix_len = min(predicted_delay_steps, previous_chunk.shape[0])
            previous_for_rtc = previous_chunk if prefix_len > 0 else None
            print(
                "RTC background inference start: "
                f"s={start_step}, remaining={previous_chunk.shape[0]}, "
                f"d={predicted_delay_steps}, prefix_len={prefix_len}, "
                f"state={_format_state(robot_state)}"
            )
            tic = time.monotonic()
            new_chunk = client.infer(
                top_image=top_image,
                wrist_left_image=wrist_left_image,
                robot_state=robot_state,
                previous_chunk=previous_for_rtc,
                prefix_len=prefix_len,
            )
        except BaseException as exc:
            with condition:
                shared.error = exc
                shared.stop = True
                condition.notify_all()
            return
        inference_elapsed_s = time.monotonic() - tic
        observed_delay_steps = max(1, _steps_from_elapsed(inference_elapsed_s, config.control_hz))
        if config.open_loop_horizon > 0:
            new_chunk = new_chunk[: config.open_loop_horizon]
        if new_chunk.shape[0] == 0:
            print("RTC background inference returned an empty chunk; keeping current chunk.")
            continue

        with condition:
            if shared.stop:
                return
            current_step = int(shared.ctx.step_index)
            new_step_index = max(0, current_step - start_step)
            if new_step_index >= new_chunk.shape[0]:
                new_step_index = new_chunk.shape[0] - 1
            shared.ctx = RTCActionContext(raw_chunk=new_chunk, step_index=new_step_index)
            delay_buffer.append(observed_delay_steps)
            condition.notify_all()

        print(
            "RTC background inference done: "
            f"latency={inference_elapsed_s:.3f}s, observed_delay={observed_delay_steps}, "
            f"new_step_index={new_step_index}, delay_buffer={list(delay_buffer)}"
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
