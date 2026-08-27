"""Input and output transforms for the NZ100 left-arm robot."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# Dataset state layout from LeRobot TCP export is 30D:
#   0-7   left joints (7), left gripper
#   8-15  right joints (7), right gripper
#   16-22 left tcp pose x/y/z/qx/qy/qz/qw
#   23-29 right tcp pose x/y/z/qx/qy/qz/qw
#
# The model-facing left-arm state layout is packed to 15D:
#   [left joints (7), left gripper, left tcp pose x/y/z/qx/qy/qz/qw].
#
# Raw left-arm action layout stays:
# [left joints (7), left gripper].
NZ100_LEFT_DATASET_STATE_INDICES: tuple[int, ...] = (*range(8), *range(16, 23))
NZ100_LEFT_PACKED_STATE_INDICES: tuple[int, ...] = tuple(range(15))
NZ100_LEFT_ACTION_INDICES: tuple[int, ...] = tuple(range(8))


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert a LeRobot image to uint8 HWC format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _validate_indices(name: str, indices: tuple[int, ...], raw_dim: int) -> None:
    if not indices:
        raise ValueError(f"NZ100 left {name} indices cannot be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"NZ100 left {name} indices contain duplicates: {indices}")
    if min(indices) < 0 or max(indices) >= raw_dim:
        raise ValueError(f"NZ100 left {name} indices {indices} are invalid for raw dimension {raw_dim}")


@dataclasses.dataclass(frozen=True)
class NZ100LeftInputs(transforms.DataTransformFn):
    """Convert raw NZ100 left-arm samples to OpenPI's common model input format."""

    model_type: _model.ModelType

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["state"], dtype=np.float32)
        state_indices = (
            NZ100_LEFT_PACKED_STATE_INDICES
            if raw_state.shape[-1] == len(NZ100_LEFT_PACKED_STATE_INDICES)
            else NZ100_LEFT_DATASET_STATE_INDICES
        )
        _validate_indices("state", state_indices, raw_state.shape[-1])

        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected NZ100 left cameras: {sorted(unexpected)}")
        if "cam_high" not in in_images:
            raise ValueError("NZ100 left input requires the cam_high image")

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
            "state": raw_state[..., state_indices],
        }

        if "actions" in data:
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            _validate_indices("action", NZ100_LEFT_ACTION_INDICES, raw_actions.shape[-1])
            result["actions"] = raw_actions[..., NZ100_LEFT_ACTION_INDICES]

        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt

        return result


@dataclasses.dataclass(frozen=True)
class NZ100LeftOutputs(transforms.DataTransformFn):
    """Return the 8 physical NZ100 left-arm action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : len(NZ100_LEFT_ACTION_INDICES)]}
