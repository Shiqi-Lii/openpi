"""Asynchronous action-queue execution runner."""

from __future__ import annotations

import collections
import concurrent.futures
import time

import numpy as np

from robot_client.config import ClientConfig
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.runners.common import format_action
from robot_client.runners.common import infer_sync_chunk
from robot_client.runners.common import read_observation
from robot_client.state_builder import NZ100RobotState
from robot_client.state_builder import discretize_plc_grippers
from robot_client.state_builder import split_action
from robot_client.sync_client import NZ100SyncClient


def run(config: ClientConfig, *, ros_io: NZ100Ros2IO | None, mock: bool, once: bool) -> None:
    if config.control_hz <= 0:
        raise ValueError(f"control_hz must be positive for async_queue, got {config.control_hz}")

    client = NZ100SyncClient(config)
    worker_client = NZ100SyncClient(config)
    action_queue: collections.deque[np.ndarray] = collections.deque()
    last_action: np.ndarray | None = None
    executed_steps = 0
    refill_threshold = max(
        0,
        min(int(config.action_refill_threshold), max(int(config.open_loop_horizon) - 1, 0)),
    )
    print(f"Entering async_queue control loop; refill_threshold={refill_threshold}.")

    first_chunk = infer_sync_chunk(client, config, ros_io, mock=mock)
    for action in first_chunk:
        action_queue.append(np.asarray(action, dtype=np.float32))
    if not action_queue:
        raise RuntimeError("Policy returned empty action chunk.")

    def submit_prefetch(
        executor: concurrent.futures.ThreadPoolExecutor,
        queued_actions: list[np.ndarray],
    ) -> concurrent.futures.Future[np.ndarray]:
        return executor.submit(
            _infer_projected_sync_chunk,
            worker_client,
            config,
            ros_io,
            queued_actions,
            mock=mock,
            log_prefix="[async-prefetch] ",
        )

    period_s = 1.0 / config.control_hz
    next_tick = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="async_queue") as executor:
        pending_future: concurrent.futures.Future[np.ndarray] | None = submit_prefetch(executor, list(action_queue))
        while True:
            if pending_future is not None and pending_future.done():
                prefetched_chunk = pending_future.result()
                for action in prefetched_chunk:
                    action_queue.append(np.asarray(action, dtype=np.float32))
                print(f"Collected async prefetch; queue_len={len(action_queue)}")
                pending_future = None

            if len(action_queue) <= refill_threshold and pending_future is None:
                print(f"Queue low ({len(action_queue)} <= {refill_threshold}); starting async prefetch.")
                pending_future = submit_prefetch(executor, list(action_queue))

            if action_queue:
                raw_action = action_queue.popleft()
                last_action = raw_action
            elif last_action is not None:
                raw_action = last_action
                print("Action queue empty; holding last action.")
            else:
                raise RuntimeError("No action available to execute.")

            action = discretize_plc_grippers(split_action(raw_action))
            print(
                f"Executing async_queue action[{executed_steps}] "
                f"queue_len={len(action_queue)}: {format_action(action)}"
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
                print("--once enabled; stopping after one async_queue action.")
                return

            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()


def _infer_projected_sync_chunk(
    client: NZ100SyncClient,
    config: ClientConfig,
    ros_io: NZ100Ros2IO | None,
    queued_actions: list[np.ndarray],
    *,
    mock: bool,
    log_prefix: str = "",
) -> np.ndarray:
    top_image, wrist_left_image, robot_state = read_observation(ros_io, mock=mock)
    if queued_actions:
        robot_state = _project_robot_state_to_queue_tail(robot_state, queued_actions[-1])
        print(f"{log_prefix}Projected state to queued tail before prefetch.")

    action_chunk = client.infer(
        top_image=top_image,
        wrist_left_image=wrist_left_image,
        robot_state=robot_state,
    )
    print(f"{log_prefix}Received action chunk: shape={tuple(action_chunk.shape)}")
    if config.open_loop_horizon > 0:
        action_chunk = action_chunk[: config.open_loop_horizon]
    return action_chunk


def _project_robot_state_to_queue_tail(robot_state: NZ100RobotState, tail_action: np.ndarray) -> NZ100RobotState:
    tail = split_action(tail_action)
    return NZ100RobotState(
        left_joints=np.asarray(tail.left_joints, dtype=np.float32),
        right_joints=np.asarray(tail.right_joints, dtype=np.float32),
        left_gripper=float(tail.left_gripper),
        right_gripper=float(tail.right_gripper),
    )
