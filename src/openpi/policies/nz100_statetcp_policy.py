"""Input and output transforms for the dual-arm NZ100 with both TCP poses."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# LeRobot TCP export layout:
# [left joints (7), left gripper, right joints (7), right gripper,
#  left TCP (x/y/z/qx/qy/qz/qw), right TCP (x/y/z/qx/qy/qz/qw)].
NZ100_STATETCP_STATE_INDICES: tuple[int, ...] = tuple(range(30))
NZ100_STATETCP_ACTION_INDICES: tuple[int, ...] = tuple(range(16))


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _validate_indices(name: str, indices: tuple[int, ...], raw_dim: int) -> None:
    if min(indices) < 0 or max(indices) >= raw_dim:
        raise ValueError(f"NZ100 state-TCP {name} requires {len(indices)} values, got {raw_dim}")


@dataclasses.dataclass(frozen=True)
class NZ100StateTCPInputs(transforms.DataTransformFn):
    """Convert 30D dual-arm NZ100 state samples to common model inputs."""

    model_type: _model.ModelType

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["state"], dtype=np.float32)
        _validate_indices("state", NZ100_STATETCP_STATE_INDICES, raw_state.shape[-1])

        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected NZ100 state-TCP cameras: {sorted(unexpected)}")
        if "cam_high" not in in_images:
            raise ValueError("NZ100 state-TCP input requires the cam_high image")

        base_image = _parse_image(in_images["cam_high"])
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for model_key, robot_key in (
            ("left_wrist_0_rgb", "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if robot_key in in_images:
                images[model_key] = _parse_image(in_images[robot_key])
                image_masks[model_key] = np.True_
            else:
                images[model_key] = np.zeros_like(base_image)
                image_masks[model_key] = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        result = {
            "image": images,
            "image_mask": image_masks,
            "state": raw_state[..., NZ100_STATETCP_STATE_INDICES],
        }
        if "actions" in data:
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            _validate_indices("action", NZ100_STATETCP_ACTION_INDICES, raw_actions.shape[-1])
            result["actions"] = raw_actions[..., NZ100_STATETCP_ACTION_INDICES]
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return result


@dataclasses.dataclass(frozen=True)
class NZ100StateTCPOutputs(transforms.DataTransformFn):
    """Return the 16 physical dual-arm joint/gripper action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : len(NZ100_STATETCP_ACTION_INDICES)]}
