"""ROS 2 IO implementation for the NZ100 robot client.

Topics are aligned with:
- /home/pc/VLA/lerobot_data_collection
- /home/pc/VLA/robot_control_pc

Only the policy-relevant signals are handled here:
top/left-wrist cameras, left/right joint positions, and PLC-style left/right gripper control.
"""

from __future__ import annotations

import threading
import time

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

        self._latest_top_image = None
        self._latest_wrist_left_image = None
        self._latest_joint_state = None
        self._latest_left_gripper = float(config.gripper_default_value)
        self._latest_right_gripper = float(config.gripper_default_value)
        self._received_left_gripper_state = False
        self._received_right_gripper_state = False

        self._left_trajectory_pub = None
        self._right_trajectory_pub = None
        self._modbus_gripper_pub = None

    def connect(self) -> None:
        print("Connecting to NZ100 ROS2 IO...")
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

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()

        class _NZ100Node(Node):
            pass

        self._node = _NZ100Node("openpi_nz100_robot_client")
        self._node.create_subscription(Image, self.config.top_camera_topic, self._on_top_image, 10)
        self._node.create_subscription(
            Image, self.config.wrist_left_camera_topic, self._on_wrist_left_image, 10
        )
        self._node.create_subscription(JointState, self.config.joint_state_topic, self._on_joint_state, 100)

        self._left_trajectory_pub = self._node.create_publisher(
            JointTrajectory, self.config.left_trajectory_topic, 10
        )
        self._right_trajectory_pub = self._node.create_publisher(
            JointTrajectory, self.config.right_trajectory_topic, 10
        )

        self._setup_gripper_io()

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(target=self._spin, daemon=True)
        self._executor_thread.start()

        print(
            "Waiting for first NZ100 observation: "
            f"top_camera={self.config.top_camera_topic}, "
            f"wrist_left_camera={self.config.wrist_left_camera_topic}, "
            f"joint_state={self.config.joint_state_topic}, "
            f"gripper_state={self.config.gripper_state_topic}"
        )
        self._wait_for_first_observation()
        print(
            "NZ100 ROS2 IO connected: "
            f"top_camera={self.config.top_camera_topic}, "
            f"wrist_left_camera={self.config.wrist_left_camera_topic}, "
            f"joint_state={self.config.joint_state_topic}, "
            f"left_traj={self.config.left_trajectory_topic}, "
            f"right_traj={self.config.right_trajectory_topic}, "
            f"gripper_cmd={self.config.gripper_cmd_topic}"
        )

    def disconnect(self) -> None:
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

    def get_wrist_left_image(self) -> np.ndarray:
        if self._latest_wrist_left_image is None:
            self._wait_for_first_observation(require_image=True, require_joint_state=False)
        return _image_msg_to_rgb(self._latest_wrist_left_image)

    def get_robot_state(self) -> NZ100RobotState:
        if self._latest_joint_state is None:
            self._wait_for_first_observation(require_image=False, require_joint_state=True)
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
        self._publish_grippers(action.left_gripper, action.right_gripper)

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
        self._publish_grippers(open_value, open_value)
        time.sleep(duration_s)
        print("NZ100 startup pose command completed; starting policy inference.")

    def _spin(self) -> None:
        while self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _setup_gripper_io(self) -> None:
        try:
            from interfaces.msg import Modbus
        except ImportError as exc:
            raise RuntimeError(
                "interfaces/msg/Modbus is required for NZ100 gripper control. "
                "Source the robot workspace that provides interfaces.msg.Modbus."
            ) from exc
        self._node.create_subscription(Modbus, self.config.gripper_state_topic, self._on_modbus_gripper_state, 100)
        self._modbus_gripper_pub = self._node.create_publisher(Modbus, self.config.gripper_cmd_topic, 10)

    def _on_top_image(self, msg) -> None:
        self._latest_top_image = msg

    def _on_wrist_left_image(self, msg) -> None:
        self._latest_wrist_left_image = msg

    def _on_joint_state(self, msg) -> None:
        self._latest_joint_state = msg

    def _on_modbus_gripper_state(self, msg) -> None:
        names = list(msg.in_out)
        values = list(msg.values)
        for index, name in enumerate(names):
            if index >= len(values):
                break
            if name == self.config.left_gripper_key:
                self._latest_left_gripper = _modbus_to_policy_value(
                    values[index], self.config.modbus_open_value, self.config.modbus_closed_value
                )
                self._received_left_gripper_state = True
            elif name == self.config.right_gripper_key:
                self._latest_right_gripper = _modbus_to_policy_value(
                    values[index], self.config.modbus_open_value, self.config.modbus_closed_value
                )
                self._received_right_gripper_state = True

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

    def _publish_grippers(self, left_value: float, right_value: float) -> None:
        self._publish_modbus_grippers(left_value, right_value)

    def _publish_modbus_grippers(self, left_value: float, right_value: float) -> None:
        from interfaces.msg import Modbus

        if self._modbus_gripper_pub is None:
            return
        msg = Modbus()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.in_out = [self.config.left_gripper_key, self.config.right_gripper_key]
        msg.values = [
            _policy_value_to_modbus(left_value, self.config.modbus_open_value, self.config.modbus_closed_value),
            _policy_value_to_modbus(right_value, self.config.modbus_open_value, self.config.modbus_closed_value),
        ]
        self._modbus_gripper_pub.publish(msg)

    def _wait_for_first_observation(
        self,
        *,
        require_image: bool = True,
        require_joint_state: bool = True,
        require_gripper_state: bool = True,
    ) -> None:
        last_status_time = 0.0
        while True:
            image_ok = (
                self._latest_top_image is not None and self._latest_wrist_left_image is not None
            ) or not require_image
            joint_ok = self._latest_joint_state is not None or not require_joint_state
            gripper_ok = (
                self._received_left_gripper_state and self._received_right_gripper_state
            ) or not require_gripper_state
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
                if require_image and self._latest_wrist_left_image is None:
                    missing.append(self.config.wrist_left_camera_topic)
                if require_joint_state and self._latest_joint_state is None:
                    missing.append(self.config.joint_state_topic)
                if require_gripper_state:
                    gripper_missing = []
                    if not self._received_left_gripper_state:
                        gripper_missing.append(self.config.left_gripper_key)
                    if not self._received_right_gripper_state:
                        gripper_missing.append(self.config.right_gripper_key)
                    if gripper_missing:
                        missing.append(f"{self.config.gripper_state_topic} keys={gripper_missing}")
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
