#!/usr/bin/env python3

import time

import rclpy
from builtin_interfaces.msg import Duration
from interfaces.msg import Modbus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

#HOME
LEFT = [0.36, 0.36, -0.01, 1.92, 1.57, 0.00, -1.40]
RIGHT = [-0.36, 0.36, -0.01, 1.92, 1.57, 0.00, 0.78]

#P1
# LEFT = [-0.06, 0.29, -0.21, 1.39, 1.76, -0.60, -1.66]
# RIGHT = [-0.36, 0.33, -0.09, 1.93, 1.42, 0.15, 0.97]

TIME_SEC = 2.0  # 到达时间（秒），支持小数，例如 2.5
LEFT_GRIPPER = "open"   # open / close
RIGHT_GRIPPER = "open"  # open / close

def trajectory(side: str, positions: list[float]) -> JointTrajectory:
    msg = JointTrajectory()
    msg.joint_names = [f"{side}_joint{i}" for i in range(1, 8)]
    point = JointTrajectoryPoint()
    point.positions = positions
    total_nanoseconds = round(TIME_SEC * 1_000_000_000)
    point.time_from_start = Duration(
        sec=total_nanoseconds // 1_000_000_000,
        nanosec=total_nanoseconds % 1_000_000_000,
    )
    msg.points = [point]
    return msg


def main() -> None:
    rclpy.init()
    node = Node("send_dual_arm_pose")
    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    left_pub = node.create_publisher(
        JointTrajectory, "/arm_left_controller/joint_trajectory", qos
    )
    right_pub = node.create_publisher(
        JointTrajectory, "/arm_right_controller/joint_trajectory", qos
    )
    gripper_pub = node.create_publisher(Modbus, "/robot/api/io/cmd", 10)

    deadline = time.monotonic() + 5.0
    while (
        left_pub.get_subscription_count() == 0
        or right_pub.get_subscription_count() == 0
        or gripper_pub.get_subscription_count() == 0
    ):
        if time.monotonic() >= deadline:
            raise RuntimeError("Arm or gripper controller subscribers not found")
        rclpy.spin_once(node, timeout_sec=0.1)

    left_msg = trajectory("left", LEFT)
    right_msg = trajectory("right", RIGHT)
    gripper_values = {"open": 1, "close": 2}
    if LEFT_GRIPPER not in gripper_values or RIGHT_GRIPPER not in gripper_values:
        raise ValueError("LEFT_GRIPPER and RIGHT_GRIPPER must be 'open' or 'close'")
    gripper_msg = Modbus()
    gripper_msg.in_out = ["an_out_d9746", "an_out_d9747"]
    gripper_msg.values = [gripper_values[LEFT_GRIPPER], gripper_values[RIGHT_GRIPPER]]
    for _ in range(10):
        left_pub.publish(left_msg)
        right_pub.publish(right_msg)
        gripper_msg.header.stamp = node.get_clock().now().to_msg()
        gripper_pub.publish(gripper_msg)
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info(
        f"Dual-arm trajectory sent ({TIME_SEC:g} s), "
        f"grippers: left={LEFT_GRIPPER}, right={RIGHT_GRIPPER}."
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
