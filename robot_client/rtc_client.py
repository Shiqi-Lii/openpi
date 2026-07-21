"""Real-Time Chunking NZ100 OpenPI policy client.

This file is intentionally separate from ``sync_client.py`` so the normal
request-response execution path stays small and unchanged.
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
class RTCContext:
    """Previous action chunk information used to guide the next chunk."""

    prev_actions: np.ndarray
    prefix_len: int
    method: str
    guidance_weight: float
    decay_tau: float


class NZ100RTCClient:
    """OpenPI websocket client with optional RTC context per inference call."""

    def __init__(self, config: ClientConfig) -> None:
        if config.execution_mode not in ("rtc_prefix", "rtc_guidance"):
            raise ValueError(
                "NZ100RTCClient requires execution_mode to be 'rtc_prefix' or 'rtc_guidance', "
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
        prompt: str | None = None,
    ) -> np.ndarray:
        """Return an RTC action chunk with shape ``(action_horizon, 16)``."""

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

        rtc_context = self._make_rtc_context(previous_chunk)
        if rtc_context is not None:
            observation["_rtc"] = dataclasses.asdict(rtc_context)

        result = self._policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)

        if actions.ndim != 2 or actions.shape[-1] != 16:
            raise ValueError(f"Expected action chunk shape (horizon, 16), got {actions.shape}")
        return actions

    def reset(self) -> None:
        self._policy.reset()

    def _make_rtc_context(self, previous_chunk: np.ndarray | None) -> RTCContext | None:
        if previous_chunk is None:
            return None

        previous_chunk = np.asarray(previous_chunk, dtype=np.float32)
        if previous_chunk.ndim != 2 or previous_chunk.shape[-1] != 16:
            raise ValueError(f"Expected previous action chunk shape (horizon, 16), got {previous_chunk.shape}")

        prefix_len = min(int(self._config.rtc_prefix_len), previous_chunk.shape[0])
        if prefix_len <= 0:
            return None

        method = "prefix" if self._config.execution_mode == "rtc_prefix" else "guidance"
        return RTCContext(
            prev_actions=build_raw_action_chunk(previous_chunk),
            prefix_len=prefix_len,
            method=method,
            guidance_weight=float(self._config.rtc_guidance_weight),
            decay_tau=float(self._config.rtc_decay_tau),
        )
