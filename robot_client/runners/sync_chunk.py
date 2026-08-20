"""Synchronous chunk execution runner."""

from __future__ import annotations

import time

from robot_client.config import ClientConfig
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.runners.common import format_action
from robot_client.runners.common import format_state
from robot_client.runners.common import read_observation
from robot_client.runtime_control import RuntimeControl
from robot_client.state_builder import discretize_plc_grippers
from robot_client.state_builder import split_action
from robot_client.sync_client import NZ100SyncClient


def run(
    config: ClientConfig,
    *,
    ros_io: NZ100Ros2IO | None,
    mock: bool,
    once: bool,
    runtime_control: RuntimeControl | None = None,
) -> None:
    client = NZ100SyncClient(config)
    executed_steps = 0
    print("Entering sync_chunk control loop.")

    while True:
        if runtime_control is not None and runtime_control.stop_event.is_set():
            print("Runtime stop requested; leaving sync_chunk loop.")
            return
        if _handle_runtime_pause(runtime_control):
            continue

        print(f"Reading observation before request; executed_steps={executed_steps}")
        top_image, robot_state = read_observation(ros_io, mock=mock)
        print(f"Requesting action chunk from OpenPI server; state={format_state(robot_state)}")
        action_chunk = client.infer(top_image=top_image, robot_state=robot_state)
        print(f"Received action chunk: shape={tuple(action_chunk.shape)}")
        if config.open_loop_horizon > 0:
            action_chunk = action_chunk[: config.open_loop_horizon]
        if not config.execute_full_chunk:
            action_chunk = action_chunk[:1]
        print(f"Executing {len(action_chunk)} actions from current chunk.")

        executed_steps, interrupted = _execute_interruptible_action_chunk(
            action_chunk,
            config=config,
            ros_io=ros_io,
            mock=mock,
            executed_steps=executed_steps,
            runtime_control=runtime_control,
        )
        if interrupted:
            continue
        if config.max_steps > 0 and executed_steps >= config.max_steps:
            print(f"Reached max_steps={config.max_steps}; stopping.")
            return
        if once:
            print("--once enabled; stopping after one chunk.")
            return


def _execute_interruptible_action_chunk(
    action_chunk,
    *,
    config: ClientConfig,
    ros_io: NZ100Ros2IO | None,
    mock: bool,
    executed_steps: int,
    runtime_control: RuntimeControl | None,
) -> tuple[int, bool]:
    step_sleep = 1.0 / config.control_hz if config.control_hz > 0 else 0.0
    for raw_action in action_chunk:
        if runtime_control is not None and runtime_control.stop_event.is_set():
            print("Runtime stop requested; leaving sync_chunk loop.")
            return executed_steps, True
        if _handle_runtime_pause(runtime_control):
            print("sync_chunk resumed; discarding remaining old chunk and requesting a fresh chunk.")
            return executed_steps, True

        action = discretize_plc_grippers(split_action(raw_action))
        print(f"Executing action[{executed_steps}]: {format_action(action)}")
        if mock:
            print(action)
        else:
            ros_io.apply_action(action)

        executed_steps += 1
        if config.max_steps > 0 and executed_steps >= config.max_steps:
            return executed_steps, False
        if step_sleep > 0:
            time.sleep(step_sleep)
    return executed_steps, False


def _handle_runtime_pause(runtime_control: RuntimeControl | None) -> bool:
    if runtime_control is None or not runtime_control.pause_event.is_set():
        return False

    print("sync_chunk pause requested; entering middle policy slot.")
    runtime_control.paused_event.set()
    if runtime_control.on_pause is not None:
        runtime_control.on_pause()
    while not runtime_control.resume_event.is_set() and not runtime_control.stop_event.is_set():
        time.sleep(0.001)
    runtime_control.pause_event.clear()
    runtime_control.resume_event.clear()
    runtime_control.paused_event.clear()
    print("sync_chunk pause finished.")
    return True
