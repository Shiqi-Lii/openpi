"""NZ100 state/action layout helpers used by the robot-side client.

This file mirrors the layout expected by ``src/openpi/policies/nz100_policy.py``
without importing the full OpenPI training/model package on the robot computer.
"""

from __future__ import annotations

import dataclasses

import numpy as np


RAW_STATE_DIM = 30

LEFT_JOINT_SLICE = slice(0, 7)
RIGHT_JOINT_SLICE = slice(14, 21)
LEFT_GRIPPER_INDEX = 28
RIGHT_GRIPPER_INDEX = 29

ACTION_DIM = 16


@dataclasses.dataclass(frozen=True)
class NZ100RobotState:
    """Minimal robot state needed by the current NZ100 OpenPI policy.

    Gripper values follow the training data / PLC convention:
    1.0 = open, 2.0 = closed.
    """

    left_joints: np.ndarray
    right_joints: np.ndarray
    left_gripper: float
    right_gripper: float


@dataclasses.dataclass(frozen=True)
class NZ100Action:
    """Decoded 16D NZ100 action.

    Gripper values follow the training data / PLC convention:
    1.0 = open, 2.0 = closed.
    """

    left_joints: np.ndarray
    left_gripper: float
    right_joints: np.ndarray
    right_gripper: float


def build_raw_state(state: NZ100RobotState) -> np.ndarray:
    """Build the raw 30D state expected by the NZ100 server transform."""

    left_joints = np.asarray(state.left_joints, dtype=np.float32)
    right_joints = np.asarray(state.right_joints, dtype=np.float32)
    if left_joints.shape != (7,):
        raise ValueError(f"left_joints must have shape (7,), got {left_joints.shape}")
    if right_joints.shape != (7,):
        raise ValueError(f"right_joints must have shape (7,), got {right_joints.shape}")

    raw_state = np.zeros((RAW_STATE_DIM,), dtype=np.float32)
    raw_state[LEFT_JOINT_SLICE] = left_joints
    raw_state[RIGHT_JOINT_SLICE] = right_joints
    raw_state[LEFT_GRIPPER_INDEX] = np.float32(state.left_gripper)
    raw_state[RIGHT_GRIPPER_INDEX] = np.float32(state.right_gripper)
    return raw_state


def build_raw_action(action: np.ndarray) -> np.ndarray:
    """Build a raw 30D action from the ordered 16D NZ100 policy action."""

    ordered_action = np.asarray(action, dtype=np.float32)
    if ordered_action.shape != (ACTION_DIM,):
        raise ValueError(f"NZ100 action must have shape ({ACTION_DIM},), got {ordered_action.shape}")

    raw_action = np.zeros((RAW_STATE_DIM,), dtype=np.float32)
    raw_action[LEFT_JOINT_SLICE] = ordered_action[0:7]
    raw_action[LEFT_GRIPPER_INDEX] = ordered_action[7]
    raw_action[RIGHT_JOINT_SLICE] = ordered_action[8:15]
    raw_action[RIGHT_GRIPPER_INDEX] = ordered_action[15]
    return raw_action


def build_raw_action_chunk(actions: np.ndarray) -> np.ndarray:
    """Build raw 30D actions from an ordered 16D action chunk."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"NZ100 action chunk must have shape (horizon, {ACTION_DIM}), got {actions.shape}")
    return np.stack([build_raw_action(action) for action in actions], axis=0)


def split_action(action: np.ndarray) -> NZ100Action:
    """Split one 16D action into left/right joint and gripper commands."""

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"NZ100 action must have shape ({ACTION_DIM},), got {action.shape}")

    return NZ100Action(
        left_joints=action[0:7],
        left_gripper=float(action[7]),
        right_joints=action[8:15],
        right_gripper=float(action[15]),
    )


def discretize_plc_grippers(
    action: NZ100Action,
    threshold: float = 1.5,
    *,
    open_value: float = 1.0,
    closed_value: float = 2.0,
) -> NZ100Action:
    """Discretize gripper outputs to the PLC convention used during training.

    The NZ100 dataset stores grippers as:
    - 1.0 = open
    - 2.0 = closed
    """

    return dataclasses.replace(
        action,
        left_gripper=closed_value if action.left_gripper >= threshold else open_value,
        right_gripper=closed_value if action.right_gripper >= threshold else open_value,
    )
