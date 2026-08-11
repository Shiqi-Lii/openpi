"""Synchronous NZ100 OpenPI policy client.

This is the normal request-response inference path provided by OpenPI:
send one observation to the GPU policy server and receive one action chunk.
"""

from __future__ import annotations

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from robot_client.config import ClientConfig
from robot_client.state_builder import NZ100RobotState
from robot_client.state_builder import build_raw_state


class NZ100SyncClient:
    """Thin wrapper around OpenPI's WebsocketClientPolicy for NZ100."""

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._policy = websocket_client_policy.WebsocketClientPolicy(
            host=config.server_host,
            port=config.server_port,
        )

    def infer(
        self,
        *,
        top_image: np.ndarray,
        robot_state: NZ100RobotState,
        prompt: str | None = None,
    ) -> np.ndarray:
        """Return an action chunk with shape ``(action_horizon, 16)``."""

        image = image_tools.resize_with_pad(top_image, self._config.image_size, self._config.image_size)
        image = image_tools.convert_to_uint8(image)

        observation = {
            "images": {
                "cam_high": image,
            },
            "state": build_raw_state(robot_state),
            "prompt": self._config.prompt if prompt is None else prompt,
        }
        result = self._policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)

        if actions.ndim != 2 or actions.shape[-1] != 16:
            raise ValueError(f"Expected action chunk shape (horizon, 16), got {actions.shape}")
        return actions

    def reset(self) -> None:
        self._policy.reset()
