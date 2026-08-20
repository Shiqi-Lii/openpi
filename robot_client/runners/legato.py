"""Legato native-continuation execution runner."""

from __future__ import annotations

import collections
import dataclasses
import threading
import time

import numpy as np

from robot_client.config import ClientConfig
from robot_client.legato_client import NZ100LegatoClient
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.runners.common import format_action
from robot_client.runners.common import format_state
from robot_client.runners.common import read_observation
from robot_client.runtime_control import RuntimeControl
from robot_client.state_builder import discretize_plc_grippers
from robot_client.state_builder import split_action


@dataclasses.dataclass
class LegatoActionContext:
    """Timing and action chunk state needed for Legato execution."""

    raw_chunk: np.ndarray
    step_index: int = 0


@dataclasses.dataclass
class LegatoSharedState:
    """Mutex-protected Legato state shared by controller and inference threads."""

    ctx: LegatoActionContext
    stop: bool = False
    error: BaseException | None = None


def run(
    config: ClientConfig,
    *,
    ros_io: NZ100Ros2IO | None,
    mock: bool,
    once: bool,
    runtime_control: RuntimeControl | None = None,
) -> None:
    client = NZ100LegatoClient(config)
    executed_steps = 0
    print(f"Entering {config.execution_mode} control loop.")
    if config.control_hz <= 0:
        raise ValueError(f"control_hz must be positive for Legato, got {config.control_hz}")

    if config.ready_signal_file is not None:
        from pathlib import Path

        Path(config.ready_signal_file).touch()
        print("Legato standby ready; ROS2 and policy server are connected.")

    if config.inference_signal_file is not None:
        from pathlib import Path

        inference_signal = Path(config.inference_signal_file)
        print(f"Waiting for initial inference signal: {inference_signal}")
        while not inference_signal.exists():
            time.sleep(0.01)
        print("Initial inference signal received.")

    current_chunk = _request_initial_chunk(config, client, ros_io, mock)

    if config.start_signal_file is not None:
        from pathlib import Path

        signal_file = Path(config.start_signal_file)
        print(f"Initial Legato chunk ready; waiting for start signal: {signal_file}")
        while not signal_file.exists():
            time.sleep(0.01)
        print("Legato start signal received.")

    condition = threading.Condition()
    shared = LegatoSharedState(ctx=LegatoActionContext(raw_chunk=current_chunk, step_index=0))
    delay_buffer = collections.deque(
        [_clamp_legato_delay_steps(int(config.legato_prefix_len), config)],
        maxlen=max(1, int(config.legato_delay_buffer_size)),
    )
    inference_thread = _start_inference_thread(config, client, ros_io, mock, shared, condition, delay_buffer)

    period_s = 1.0 / config.control_hz
    next_tick = time.monotonic()
    try:
        while True:
            if runtime_control is not None and runtime_control.stop_event.is_set():
                print("Runtime stop requested; leaving Legato loop.")
                return
            if runtime_control is not None and runtime_control.pause_event.is_set():
                print("Legato pause requested; stopping current continuation and entering middle policy slot.")
                with condition:
                    shared.stop = True
                    condition.notify_all()
                inference_thread.join(timeout=2.0)

                runtime_control.paused_event.set()
                if runtime_control.on_pause is not None:
                    runtime_control.on_pause()
                while not runtime_control.resume_event.is_set() and not runtime_control.stop_event.is_set():
                    time.sleep(0.001)
                runtime_control.pause_event.clear()
                runtime_control.resume_event.clear()
                runtime_control.paused_event.clear()
                if runtime_control.stop_event.is_set():
                    print("Runtime stop requested during Legato pause.")
                    return

                print("Legato resumed; requesting a fresh initial chunk.")
                current_chunk = _request_initial_chunk(config, client, ros_io, mock)
                with condition:
                    shared = LegatoSharedState(ctx=LegatoActionContext(raw_chunk=current_chunk, step_index=0))
                delay_buffer = collections.deque(
                    [_clamp_legato_delay_steps(int(config.legato_prefix_len), config)],
                    maxlen=max(1, int(config.legato_delay_buffer_size)),
                )
                inference_thread = _start_inference_thread(
                    config, client, ros_io, mock, shared, condition, delay_buffer
                )
                next_tick = time.monotonic()
                continue

            with condition:
                if shared.error is not None:
                    raise RuntimeError("Legato background inference failed") from shared.error
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
                    print("Legato chunk exhausted before replacement; holding last action.")

            action = discretize_plc_grippers(split_action(raw_action))
            print(
                f"Executing Legato action[{executed_steps}] "
                f"chunk_step={local_step_index}: {format_action(action)}"
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
                print("--once enabled; stopping after one Legato action.")
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


def _request_initial_chunk(
    config: ClientConfig,
    client: NZ100LegatoClient,
    ros_io: NZ100Ros2IO | None,
    mock: bool,
) -> np.ndarray:
    print("Reading initial observation for Legato.")
    top_image, robot_state = read_observation(ros_io, mock=mock)
    print(f"Requesting initial Legato action chunk; state={format_state(robot_state)}")
    inference_start_s = time.monotonic()
    current_chunk = client.infer(top_image=top_image, robot_state=robot_state)
    inference_elapsed_s = time.monotonic() - inference_start_s
    print(
        "Received initial Legato chunk: "
        f"shape={tuple(current_chunk.shape)}, latency={inference_elapsed_s:.3f}s"
    )
    if config.open_loop_horizon > 0:
        current_chunk = current_chunk[: config.open_loop_horizon]
    if current_chunk.shape[0] == 0:
        raise ValueError("Initial Legato action chunk is empty")
    return current_chunk


def _start_inference_thread(
    config: ClientConfig,
    client: NZ100LegatoClient,
    ros_io: NZ100Ros2IO | None,
    mock: bool,
    shared: LegatoSharedState,
    condition: threading.Condition,
    delay_buffer: collections.deque[int],
) -> threading.Thread:
    inference_thread = threading.Thread(
        target=_legato_inference_loop,
        args=(config, client, ros_io, mock, shared, condition, delay_buffer),
        daemon=True,
    )
    inference_thread.start()
    return inference_thread


def _steps_from_elapsed(elapsed_s: float, control_hz: float) -> int:
    if control_hz <= 0:
        return 0
    return max(0, int(elapsed_s * control_hz))


def _max_legato_delay_steps(config: ClientConfig) -> int:
    if config.legato_max_delay_steps is not None:
        return max(1, int(config.legato_max_delay_steps))
    horizon = int(config.open_loop_horizon)
    if horizon <= 0:
        horizon = 50
    s_min = max(1, int(config.legato_execute_horizon))
    return max(1, horizon - s_min)


def _clamp_legato_delay_steps(delay_steps: int, config: ClientConfig) -> int:
    return min(max(1, int(delay_steps)), _max_legato_delay_steps(config))


def _legato_ramp_end(config: ClientConfig, prefix_len: int) -> int:
    horizon = int(config.open_loop_horizon)
    if horizon <= 0:
        horizon = 50
    ramp_end = horizon if config.legato_ramp_end is None else int(config.legato_ramp_end)
    return min(max(prefix_len, ramp_end), horizon)


def _legato_inference_loop(
    config: ClientConfig,
    client: NZ100LegatoClient,
    ros_io: NZ100Ros2IO | None,
    mock: bool,
    shared: LegatoSharedState,
    condition: threading.Condition,
    delay_buffer: collections.deque[int],
) -> None:
    """Background inference loop for Legato native continuation."""

    minimum_execution_horizon = max(1, int(config.legato_execute_horizon))
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
            predicted_delay_steps = _clamp_legato_delay_steps(
                max(max(delay_buffer), int(config.legato_prefix_len)), config
            )

        try:
            top_image, robot_state = read_observation(ros_io, mock=mock)
            prefix_len = min(predicted_delay_steps, previous_chunk.shape[0])
            previous_for_legato = previous_chunk if prefix_len > 0 else None
            ramp_end = _legato_ramp_end(config, prefix_len)
            print(
                "Legato background inference start: "
                f"s={start_step}, remaining={previous_chunk.shape[0]}, "
                f"d={predicted_delay_steps}, prefix_len={prefix_len}, ramp_end={ramp_end}, "
                f"state={format_state(robot_state)}"
            )
            tic = time.monotonic()
            new_chunk = client.infer(
                top_image=top_image,
                robot_state=robot_state,
                previous_chunk=previous_for_legato,
                prefix_len=prefix_len,
                ramp_end=ramp_end,
            )
        except BaseException as exc:
            with condition:
                shared.error = exc
                shared.stop = True
                condition.notify_all()
            return
        inference_elapsed_s = time.monotonic() - tic
        raw_observed_delay_steps = max(1, _steps_from_elapsed(inference_elapsed_s, config.control_hz))
        observed_delay_steps = _clamp_legato_delay_steps(raw_observed_delay_steps, config)
        if config.open_loop_horizon > 0:
            new_chunk = new_chunk[: config.open_loop_horizon]
        if new_chunk.shape[0] == 0:
            print("Legato background inference returned an empty chunk; keeping current chunk.")
            continue

        with condition:
            if shared.stop:
                return
            current_step = int(shared.ctx.step_index)
            new_step_index = max(0, current_step - start_step)
            if new_step_index >= new_chunk.shape[0]:
                new_step_index = new_chunk.shape[0] - 1
            shared.ctx = LegatoActionContext(raw_chunk=new_chunk, step_index=new_step_index)
            delay_buffer.append(observed_delay_steps)
            condition.notify_all()

        print(
            "Legato background inference done: "
            f"latency={inference_elapsed_s:.3f}s, "
            f"observed_delay={observed_delay_steps}, raw_observed_delay={raw_observed_delay_steps}, "
            f"new_step_index={new_step_index}, delay_buffer={list(delay_buffer)}"
        )
