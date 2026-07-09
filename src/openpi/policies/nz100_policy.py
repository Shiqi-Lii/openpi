"""Input and output transforms for the NZ100 dual-arm robot."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# Raw LeRobot layout, reordered to:
# [left joints (7), left gripper, right joints (7), right gripper].
NZ100_STATE_INDICES: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    28,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    29,
)
NZ100_ACTION_INDICES: tuple[int, ...] = NZ100_STATE_INDICES


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
        raise ValueError(f"NZ100 {name} indices cannot be empty")
    if len(set(indices)) != len(indices):
        raise ValueError(f"NZ100 {name} indices contain duplicates: {indices}")
    if min(indices) < 0 or max(indices) >= raw_dim:
        raise ValueError(f"NZ100 {name} indices {indices} are invalid for raw dimension {raw_dim}")


@dataclasses.dataclass(frozen=True)
class NZ100Inputs(transforms.DataTransformFn):
    """Convert raw NZ100 samples to OpenPI's common model input format."""

    model_type: _model.ModelType

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["state"], dtype=np.float32)
        _validate_indices("state", NZ100_STATE_INDICES, raw_state.shape[-1])
        if len(NZ100_STATE_INDICES) < len(NZ100_ACTION_INDICES):
            raise ValueError("NZ100 state selection must be at least as long as the action selection")
        if len(NZ100_STATE_INDICES) > 32:
            raise ValueError(f"NZ100 selects {len(NZ100_STATE_INDICES)} state values; pi0.5 supports at most 32")

        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(f"Unexpected NZ100 cameras: {sorted(unexpected)}")
        if "cam_high" not in in_images:
            raise ValueError("NZ100 input requires the cam_high image")

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
            "state": raw_state[..., NZ100_STATE_INDICES],
        }

        if "actions" in data:
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            _validate_indices("action", NZ100_ACTION_INDICES, raw_actions.shape[-1])
            result["actions"] = raw_actions[..., NZ100_ACTION_INDICES]

        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt

        return result


@dataclasses.dataclass(frozen=True)
class NZ100Outputs(transforms.DataTransformFn):
    """Return the 16 physical NZ100 action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : len(NZ100_ACTION_INDICES)]}
