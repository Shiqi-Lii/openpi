import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # Optional Legato-style native continuation for flow action chunking.
    # Disabled by default so existing training and inference behavior is unchanged.
    legato_enabled: bool = False
    # Index of an unused action dimension used to carry the per-horizon schedule.
    # Keeping this inside the existing action vector avoids changing checkpoint shapes.
    legato_omega_dim: int | None = None
    # Number of leading action dimensions that correspond to real robot actions.
    # Loss is computed on these dimensions, excluding legato_omega_dim if it is inside this range.
    legato_loss_action_dim: int | None = None
    # Denoising step count used to construct the Legato training target.
    legato_train_num_steps: int = 10
    # Fixed deployment schedule: full-guidance prefix length d and linear ramp length r.
    legato_full_guidance_steps: int = 0
    legato_ramp_steps: int = 0
    # Optional schedule randomization, following the paper's d/r randomization idea.
    legato_randomize_schedule: bool = False
    legato_full_guidance_min: int = 0
    legato_full_guidance_max: int = 0
    legato_ramp_min: int = 0
    legato_ramp_max: int = 0

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.legato_enabled:
            if self.legato_omega_dim is None:
                raise ValueError("legato_omega_dim must be set when legato_enabled=True")
            if not 0 <= self.legato_omega_dim < self.action_dim:
                raise ValueError(
                    f"legato_omega_dim must be in [0, {self.action_dim}), got {self.legato_omega_dim}"
                )
            if self.legato_train_num_steps <= 0:
                raise ValueError(f"legato_train_num_steps must be positive, got {self.legato_train_num_steps}")
            if self.legato_loss_action_dim is not None and not 0 < self.legato_loss_action_dim <= self.action_dim:
                raise ValueError(
                    f"legato_loss_action_dim must be in (0, {self.action_dim}], got {self.legato_loss_action_dim}"
                )
            for name in (
                "legato_full_guidance_steps",
                "legato_ramp_steps",
                "legato_full_guidance_min",
                "legato_full_guidance_max",
                "legato_ramp_min",
                "legato_ramp_max",
            ):
                if getattr(self, name) < 0:
                    raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
            if self.legato_randomize_schedule:
                if self.legato_full_guidance_max < self.legato_full_guidance_min:
                    raise ValueError("legato_full_guidance_max must be >= legato_full_guidance_min")
                if self.legato_ramp_max < self.legato_ramp_min:
                    raise ValueError("legato_ramp_max must be >= legato_ramp_min")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
