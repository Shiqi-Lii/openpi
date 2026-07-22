"""Real-Time Chunking guidance execution runner."""

from __future__ import annotations

import collections
import dataclasses
import threading
import time

import numpy as np

from robot_client.config import ClientConfig
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.rtc_client import NZ100RTCClient
from robot_client.runners.common import format_action
from robot_client.runners.common import format_state
from robot_client.runners.common import read_observation
from robot_client.state_builder import discretize_plc_grippers
from robot_client.state_builder import split_action


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


def run(config: ClientConfig, *, ros_io: NZ100Ros2IO | None, mock: bool, once: bool) -> None:
    client = NZ100RTCClient(config)
    executed_steps = 0
    print(f"Entering {config.execution_mode} control loop.")
    if config.control_hz <= 0:
        raise ValueError(f"control_hz must be positive for RTC, got {config.control_hz}")

    print("Reading initial observation for RTC.")
    top_image, wrist_left_image, robot_state = read_observation(ros_io, mock=mock)
    print(f"Requesting initial RTC action chunk; state={format_state(robot_state)}")
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
        [_clamp_rtc_delay_steps(int(config.rtc_prefix_len), config)],
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


def _steps_from_elapsed(elapsed_s: float, control_hz: float) -> int:
    if control_hz <= 0:
        return 0
    return max(0, int(elapsed_s * control_hz))


def _max_rtc_delay_steps(config: ClientConfig) -> int:
    if config.rtc_max_delay_steps is not None:
        return max(1, int(config.rtc_max_delay_steps))
    horizon = int(config.open_loop_horizon)
    if horizon <= 0:
        horizon = 50
    s_min = max(1, int(config.rtc_execute_horizon))
    return max(1, horizon - s_min)


def _clamp_rtc_delay_steps(delay_steps: int, config: ClientConfig) -> int:
    return min(max(1, int(delay_steps)), _max_rtc_delay_steps(config))


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
            predicted_delay_steps = _clamp_rtc_delay_steps(max(max(delay_buffer), int(config.rtc_prefix_len)), config)

        try:
            top_image, wrist_left_image, robot_state = read_observation(ros_io, mock=mock)
            prefix_len = min(predicted_delay_steps, previous_chunk.shape[0])
            previous_for_rtc = previous_chunk if prefix_len > 0 else None
            print(
                "RTC background inference start: "
                f"s={start_step}, remaining={previous_chunk.shape[0]}, "
                f"d={predicted_delay_steps}, prefix_len={prefix_len}, "
                f"state={format_state(robot_state)}"
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
        raw_observed_delay_steps = max(1, _steps_from_elapsed(inference_elapsed_s, config.control_hz))
        observed_delay_steps = _clamp_rtc_delay_steps(raw_observed_delay_steps, config)
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
            f"latency={inference_elapsed_s:.3f}s, "
            f"observed_delay={observed_delay_steps}, raw_observed_delay={raw_observed_delay_steps}, "
            f"new_step_index={new_step_index}, delay_buffer={list(delay_buffer)}"
        )

