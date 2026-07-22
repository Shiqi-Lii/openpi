"""Synchronous chunk execution runner."""

from __future__ import annotations

from robot_client.config import ClientConfig
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.runners.common import execute_action_chunk
from robot_client.runners.common import format_state
from robot_client.runners.common import read_observation
from robot_client.sync_client import NZ100SyncClient


def run(config: ClientConfig, *, ros_io: NZ100Ros2IO | None, mock: bool, once: bool) -> None:
    client = NZ100SyncClient(config)
    executed_steps = 0
    print("Entering sync_chunk control loop.")

    while True:
        print(f"Reading observation before request; executed_steps={executed_steps}")
        top_image, wrist_left_image, robot_state = read_observation(ros_io, mock=mock)
        print(f"Requesting action chunk from OpenPI server; state={format_state(robot_state)}")
        action_chunk = client.infer(
            top_image=top_image, wrist_left_image=wrist_left_image, robot_state=robot_state
        )
        print(f"Received action chunk: shape={tuple(action_chunk.shape)}")
        if config.open_loop_horizon > 0:
            action_chunk = action_chunk[: config.open_loop_horizon]
        if not config.execute_full_chunk:
            action_chunk = action_chunk[:1]
        print(f"Executing {len(action_chunk)} actions from current chunk.")

        executed_steps = execute_action_chunk(
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

