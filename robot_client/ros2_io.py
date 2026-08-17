"""Robot IO implementation for the NZ100 robot client.

Topics are aligned with:
- /home/pc/VLA/lerobot_data_collection
- /home/pc/VLA/robot_control_pc

Only the policy-relevant signals are handled here:
top camera, left/right joint positions, and PLC-style left/right gripper control.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

from robot_client.config import Ros2Config
from robot_client.state_builder import NZ100Action
from robot_client.state_builder import NZ100RobotState


class NZ100Ros2IO:
    """Subscribe robot observations and publish low-level NZ100 commands."""

    def __init__(self, config: Ros2Config) -> None:
        self.config = config
        self._rclpy = None
        self._executor = None
        self._executor_thread: threading.Thread | None = None
        self._node = None
        self._robot = None
        self._arm_type = None

        self._latest_top_image = None
        self._latest_joint_state = None
        self._latest_left_gripper = float(config.gripper_default_value)
        self._latest_right_gripper = float(config.gripper_default_value)
        self._last_left_gripper_cmd: int | None = None
        self._last_right_gripper_cmd: int | None = None

        self._left_trajectory_pub = None
        self._right_trajectory_pub = None

    def connect(self) -> None:
        print("Connecting to NZ100 camera ROS2 topic and YSRobot SDK...")
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from sensor_msgs.msg import Image, JointState
            from trajectory_msgs.msg import JointTrajectory
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 dependencies are not available. Source ROS 2 and the robot workspace before running."
            ) from exc

        sdk_path = str(Path(self.config.ysrobot_sdk_path).expanduser())
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        try:
            from ysrobot import ArmType, RobotClient
        except ImportError as exc:
            raise RuntimeError(
                f"YSRobot SDK is not available from {sdk_path}. Check ysrobot_sdk_path in the client config."
            ) from exc

        self._rclpy = rclpy
        self._arm_type = ArmType
        self._robot = RobotClient(
            self.config.ysrobot_host,
            port=int(self.config.ysrobot_port),
            timeout_ms=int(self.config.ysrobot_timeout_ms),
        )

        result = self._robot.login(self.config.ysrobot_login_level, self.config.ysrobot_login_pin)
        print("YSRobot login:", result.success, result.message)
        if not result:
            raise RuntimeError(f"YSRobot login failed: {result.message}")

        result = self._robot.connect()
        print("YSRobot connect:", result.success, result.message)
        if not result:
            raise RuntimeError(f"YSRobot connect failed: {result.message}")

        if not rclpy.ok():
            rclpy.init()

        class _NZ100Node(Node):
            pass

        self._node = _NZ100Node("openpi_nz100_robot_client")
        self._node.create_subscription(Image, self.config.top_camera_topic, self._on_top_image, 10)
        self._node.create_subscription(JointState, self.config.joint_state_topic, self._on_joint_state, 100)

        self._left_trajectory_pub = self._node.create_publisher(
            JointTrajectory, self.config.left_trajectory_topic, 10
        )
        self._right_trajectory_pub = self._node.create_publisher(
            JointTrajectory, self.config.right_trajectory_topic, 10
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(target=self._spin, daemon=True)
        self._executor_thread.start()

        print(
            "Waiting for first NZ100 observation: "
            f"top_camera={self.config.top_camera_topic}, "
            f"joint_state={self.config.joint_state_topic}, "
            f"ysrobot={self.config.ysrobot_host}:{self.config.ysrobot_port}"
        )
        self._wait_for_first_observation(require_gripper_state=False)
        print(
            "NZ100 IO connected: "
            f"top_camera={self.config.top_camera_topic}, "
            f"joint_state={self.config.joint_state_topic}, "
            f"left_traj={self.config.left_trajectory_topic}, "
            f"right_traj={self.config.right_trajectory_topic}, "
            f"ysrobot={self.config.ysrobot_host}:{self.config.ysrobot_port}"
        )

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        print("NZ100 ROS2 IO disconnected")

    def get_top_image(self) -> np.ndarray:
        if self._latest_top_image is None:
            self._wait_for_first_observation(require_image=True, require_joint_state=False)
        return _image_msg_to_rgb(self._latest_top_image)

    def get_robot_state(self) -> NZ100RobotState:
        if self._latest_joint_state is None:
            self._wait_for_first_observation(require_image=False, require_joint_state=True, require_gripper_state=False)
        return NZ100RobotState(
            left_joints=self._extract_named_positions(self.config.left_joint_names),
            right_joints=self._extract_named_positions(self.config.right_joint_names),
            left_gripper=float(self._latest_left_gripper),
            right_gripper=float(self._latest_right_gripper),
        )

    def apply_action(self, action: NZ100Action) -> None:
        self._publish_joint_trajectory(
            self._left_trajectory_pub,
            list(self.config.left_joint_names),
            np.asarray(action.left_joints, dtype=np.float64),
        )
        self._publish_joint_trajectory(
            self._right_trajectory_pub,
            list(self.config.right_joint_names),
            np.asarray(action.right_joints, dtype=np.float64),
        )
        self._control_grippers(action.left_gripper, action.right_gripper)

    def set_cached_gripper_state(self, *, left: float | None = None, right: float | None = None) -> None:
        """Update cached PLC gripper state after an out-of-band command.

        Multi-stage control may command the gripper directly through the SDK
        while VLA is paused. The policy observation uses this cache, so keep it
        aligned with the real gripper before resuming VLA.
        """

        if left is not None:
            left_command = int(left)
            self._latest_left_gripper = float(left_command)
            self._last_left_gripper_cmd = left_command
        if right is not None:
            right_command = int(right)
            self._latest_right_gripper = float(right_command)
            self._last_right_gripper_cmd = right_command

    def hold_current_joint_positions(self, *, duration_s: float = 0.1) -> None:
        """Publish a short hold command at the measured current joint positions.

        Multi-stage control hands the left arm from the streaming VLA trajectory
        publisher to the SDK ``move_l`` planner. Sending one measured-current
        joint target first avoids handing over from a still-moving short-horizon
        VLA target, which can otherwise cause a small twitch at the first
        ``move_l``.
        """

        if self._latest_joint_state is None:
            self._wait_for_first_observation(require_image=False, require_joint_state=True, require_gripper_state=False)
        left_positions = self._extract_named_positions(self.config.left_joint_names).astype(np.float64)
        right_positions = self._extract_named_positions(self.config.right_joint_names).astype(np.float64)
        print(f"Holding current joint positions before SDK move_l handoff: duration={duration_s:.2f}s")
        self._publish_joint_trajectory(
            self._left_trajectory_pub,
            list(self.config.left_joint_names),
            left_positions,
            duration_s=duration_s,
        )
        self._publish_joint_trajectory(
            self._right_trajectory_pub,
            list(self.config.right_joint_names),
            right_positions,
            duration_s=duration_s,
        )
        time.sleep(max(float(duration_s), 0.0))

    def move_to_home(self) -> None:
        """Move both arms to the configured startup pose before inference."""
        left_positions = np.asarray(self.config.left_home_positions, dtype=np.float64)
        right_positions = np.asarray(self.config.right_home_positions, dtype=np.float64)
        duration_s = float(self.config.home_time_from_start)
        if duration_s <= 0:
            raise ValueError("Home trajectory time must be positive")

        print(f"Moving NZ100 to startup pose: both arms={duration_s:.2f}s, opening both grippers")
        self._publish_joint_trajectory(
            self._left_trajectory_pub,
            list(self.config.left_joint_names),
            left_positions,
            duration_s=duration_s,
        )
        self._publish_joint_trajectory(
            self._right_trajectory_pub,
            list(self.config.right_joint_names),
            right_positions,
            duration_s=duration_s,
        )
        open_value = float(self.config.modbus_open_value)
        self._control_grippers(open_value, open_value)
        time.sleep(duration_s)
        print("NZ100 startup pose command completed; starting policy inference.")

    def _spin(self) -> None:
        while self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_top_image(self, msg) -> None:
        self._latest_top_image = msg

    def _on_joint_state(self, msg) -> None:
        self._latest_joint_state = msg

    def _extract_named_positions(self, joint_names: tuple[str, ...]) -> np.ndarray:
        msg = self._latest_joint_state
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        missing = [name for name in joint_names if name not in name_to_index]
        if missing:
            raise KeyError(
                f"Joint names {missing} not found in {self.config.joint_state_topic}. "
                f"Available names: {list(msg.name)}"
            )
        return np.asarray([msg.position[name_to_index[name]] for name in joint_names], dtype=np.float32)

    def _publish_joint_trajectory(
        self,
        publisher,
        joint_names: list[str],
        positions: np.ndarray,
        *,
        duration_s: float | None = None,
    ) -> None:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        if positions.shape != (len(joint_names),):
            raise ValueError(f"Expected {len(joint_names)} joint positions, got shape {positions.shape}")

        msg = JointTrajectory()
        msg.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        duration_s = float(self.config.point_time_from_start if duration_s is None else duration_s)
        if duration_s <= 0:
            raise ValueError(f"Trajectory duration must be positive, got {duration_s}")
        point.time_from_start = Duration(
            sec=int(duration_s),
            nanosec=int((duration_s % 1.0) * 1e9),
        )
        msg.points = [point]
        publisher.publish(msg)

    def _control_grippers(self, left_value: float, right_value: float) -> None:
        left_command = _policy_value_to_modbus(left_value, self.config.modbus_open_value, self.config.modbus_closed_value)
        right_command = _policy_value_to_modbus(
            right_value, self.config.modbus_open_value, self.config.modbus_closed_value
        )
        if left_command != self._last_left_gripper_cmd:
            left_result = self._robot.device.write_modbus(int(self.config.left_gripper_modbus_address), left_command)
            if not left_result:
                print(f"Warning: YSRobot left gripper Modbus command failed: {left_result.message}")
            else:
                self._last_left_gripper_cmd = left_command
                self._latest_left_gripper = float(left_command)
        if right_command != self._last_right_gripper_cmd:
            right_result = self._robot.device.write_modbus(int(self.config.right_gripper_modbus_address), right_command)
            if not right_result:
                print(f"Warning: YSRobot right gripper Modbus command failed: {right_result.message}")
            else:
                self._last_right_gripper_cmd = right_command
                self._latest_right_gripper = float(right_command)

    def _wait_for_first_observation(
        self,
        *,
        require_image: bool = True,
        require_joint_state: bool = True,
        require_gripper_state: bool = True,
    ) -> None:
        last_status_time = 0.0
        while True:
            image_ok = self._latest_top_image is not None or not require_image
            joint_ok = self._latest_joint_state is not None or not require_joint_state
            gripper_ok = True  # Gripper state uses the last commanded/default policy value.
            if image_ok and joint_ok and gripper_ok:
                print(
                    "First NZ100 observation received: "
                    f"image={'ok' if image_ok else 'skipped'}, "
                    f"joint_state={'ok' if joint_ok else 'skipped'}, "
                    f"gripper_state={'ok' if gripper_ok else 'skipped'}"
                )
                return
            now = time.time()
            if now - last_status_time >= 2.0:
                missing = []
                if require_image and self._latest_top_image is None:
                    missing.append(self.config.top_camera_topic)
                if require_joint_state and self._latest_joint_state is None:
                    missing.append(self.config.joint_state_topic)
                print(f"Waiting for ROS2 topics: {missing}")
                last_status_time = now
            time.sleep(0.05)


def _image_msg_to_rgb(msg) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    encoding = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    channels_by_encoding = {
        "mono8": 1,
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "yuv422_yuy2": 2,
        "yuyv": 2,
        "yuy2": 2,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported ROS image encoding: {msg.encoding}")

    channels = channels_by_encoding[encoding]
    row_bytes = width * channels
    if int(msg.step) < row_bytes:
        raise ValueError(f"Invalid image step={msg.step}, expected at least {row_bytes}")
    expected_bytes = height * int(msg.step)
    if data.size < expected_bytes:
        raise ValueError(f"Image data too short: {data.size} < {expected_bytes}")

    rows = data[:expected_bytes].reshape(height, int(msg.step))
    image = rows[:, :row_bytes].reshape(height, width, channels)
    if encoding == "rgb8":
        return image.copy()
    if encoding == "bgr8":
        return image[:, :, ::-1].copy()
    if encoding == "mono8":
        return np.repeat(image, 3, axis=2)
    if encoding == "rgba8":
        return image[:, :, :3].copy()
    if encoding == "bgra8":
        return image[:, :, :3][:, :, ::-1].copy()
    return _yuyv_to_rgb(image, height, width)


def _yuyv_to_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    yuyv = image.reshape(height, width // 2, 4).astype(np.float32)
    y0 = yuyv[:, :, 0]
    u = yuyv[:, :, 1]
    y1 = yuyv[:, :, 2]
    v = yuyv[:, :, 3]

    y = np.empty((height, width), dtype=np.float32)
    y[:, 0::2] = y0
    y[:, 1::2] = y1
    u_full = np.repeat(u[:, :, np.newaxis], 2, axis=2).reshape(height, width) - 128.0
    v_full = np.repeat(v[:, :, np.newaxis], 2, axis=2).reshape(height, width) - 128.0
    c = y - 16.0

    r = 1.164 * c + 1.596 * v_full
    g = 1.164 * c - 0.392 * u_full - 0.813 * v_full
    b = 1.164 * c + 2.017 * u_full
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def _modbus_to_policy_value(value, open_value: int, closed_value: int) -> float:
    value = int(value)
    if value == int(closed_value):
        return float(closed_value)
    if value == int(open_value):
        return float(open_value)
    return float(value)


def _policy_value_to_modbus(value: float, open_value: int, closed_value: int) -> int:
    return int(closed_value if _is_closed_policy_value(value) else open_value)


def _is_closed_policy_value(value: float) -> bool:
    # Training data uses 1=open, 2=closed, so the midpoint is 1.5.
    return float(value) >= 1.5
