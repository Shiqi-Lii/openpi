"""Legato NZ100 OpenPI policy client.

This client is separate from RTC so native-continuation inference can evolve
without changing the existing RTC guidance request path.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from robot_client.config import ClientConfig
from robot_client.state_builder import NZ100RobotState
from robot_client.state_builder import build_raw_action_chunk
from robot_client.state_builder import build_raw_state


@dataclasses.dataclass(frozen=True)
class LegatoContext:
    """Previous action chunk information used for Legato continuation."""

    prev_actions: np.ndarray
    prefix_len: int
    ramp_end: int


class NZ100LegatoClient:
    """OpenPI websocket client with optional Legato context per inference call."""

    def __init__(self, config: ClientConfig) -> None:
        if config.execution_mode != "legato":
            raise ValueError(
                "NZ100LegatoClient requires execution_mode to be 'legato', "
                f"got {config.execution_mode!r}"
            )
        self._config = config
        self._policy = websocket_client_policy.WebsocketClientPolicy(
            host=config.server_host,
            port=config.server_port,
        )

    def infer(
        self,
        *,
        top_image: np.ndarray,
        wrist_left_image: np.ndarray,
        robot_state: NZ100RobotState,
        previous_chunk: np.ndarray | None = None,
        prefix_len: int | None = None,
        ramp_end: int | None = None,
        prompt: str | None = None,
    ) -> np.ndarray:
        """Return a Legato action chunk with shape ``(action_horizon, 16)``."""

        image = image_tools.resize_with_pad(top_image, self._config.image_size, self._config.image_size)
        image = image_tools.convert_to_uint8(image)
        wrist_left_image = image_tools.resize_with_pad(
            wrist_left_image, self._config.image_size, self._config.image_size
        )
        wrist_left_image = image_tools.convert_to_uint8(wrist_left_image)

        observation = {
            "images": {
                "cam_high": image,
                "cam_left_wrist": wrist_left_image,
            },
            "state": build_raw_state(robot_state),
            "prompt": self._config.prompt if prompt is None else prompt,
        }

        legato_context = self._make_legato_context(previous_chunk, prefix_len=prefix_len, ramp_end=ramp_end)
        if legato_context is not None:
            observation["_legato"] = dataclasses.asdict(legato_context)

        result = self._policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)

        if actions.ndim != 2 or actions.shape[-1] != 16:
            raise ValueError(f"Expected action chunk shape (horizon, 16), got {actions.shape}")
        return actions

    def reset(self) -> None:
        self._policy.reset()

    def _make_legato_context(
        self,
        previous_chunk: np.ndarray | None,
        *,
        prefix_len: int | None = None,
        ramp_end: int | None = None,
    ) -> LegatoContext | None:
        if previous_chunk is None:
            return None

        previous_chunk = np.asarray(previous_chunk, dtype=np.float32)
        if previous_chunk.ndim != 2 or previous_chunk.shape[-1] != 16:
            raise ValueError(f"Expected previous action chunk shape (horizon, 16), got {previous_chunk.shape}")

        prefix_len = int(self._config.legato_prefix_len if prefix_len is None else prefix_len)
        prefix_len = min(prefix_len, previous_chunk.shape[0])
        if prefix_len <= 0:
            return None

        target_horizon = int(self._config.open_loop_horizon)
        if target_horizon <= 0:
            target_horizon = previous_chunk.shape[0]
        if previous_chunk.shape[0] < target_horizon:
            pad_len = target_horizon - previous_chunk.shape[0]
            pad_action = previous_chunk[-1:]
            previous_chunk = np.concatenate(
                [previous_chunk, np.repeat(pad_action, pad_len, axis=0)],
                axis=0,
            )
        elif previous_chunk.shape[0] > target_horizon:
            previous_chunk = previous_chunk[:target_horizon]

        ramp_end = self._config.legato_ramp_end if ramp_end is None else ramp_end
        if ramp_end is None:
            ramp_end = target_horizon

        return LegatoContext(
            prev_actions=build_raw_action_chunk(previous_chunk),
            prefix_len=prefix_len,
            ramp_end=int(ramp_end),
        )
