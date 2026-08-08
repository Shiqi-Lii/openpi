#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Duration
from interfaces.msg import Modbus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def make_trajectory(side: str, positions: list[float], duration: float) -> JointTrajectory:
    if len(positions) != 7 or duration <= 0:
        raise ValueError("plan requires 7 joint positions and duration > 0")
    nanoseconds = round(duration * 1_000_000_000)
    point = JointTrajectoryPoint()
    point.positions = [float(value) for value in positions]
    point.time_from_start = Duration(
        sec=nanoseconds // 1_000_000_000,
        nanosec=nanoseconds % 1_000_000_000,
    )
    msg = JointTrajectory()
    msg.joint_names = [f"{side}_joint{i}" for i in range(1, 8)]
    msg.points = [point]
    return msg


class Planner:
    def __init__(self) -> None:
        rclpy.init()
        self.node = Node("multi_stage_control")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.left = self.node.create_publisher(
            JointTrajectory, "/arm_left_controller/joint_trajectory", qos
        )
        self.right = self.node.create_publisher(
            JointTrajectory, "/arm_right_controller/joint_trajectory", qos
        )
        self.grippers = self.node.create_publisher(Modbus, "/robot/api/io/cmd", 10)

    def gripper_message(self, stage: dict) -> Modbus | None:
        if "left_gripper" not in stage and "right_gripper" not in stage:
            return None
        if "left_gripper" not in stage or "right_gripper" not in stage:
            raise ValueError("plan must set both left_gripper and right_gripper")
        values = {"open": 1, "close": 2}
        left = stage["left_gripper"]
        right = stage["right_gripper"]
        if left not in values or right not in values:
            raise ValueError("left_gripper/right_gripper must be 'open' or 'close'")
        msg = Modbus()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.in_out = ["an_out_d9746", "an_out_d9747"]
        msg.values = [values[left], values[right]]
        return msg

    def move(self, stage: dict, prefetch=None, prefetch_before: float = 0.0) -> None:
        gripper_msg = self.gripper_message(stage)
        deadline = time.monotonic() + 5.0
        while (
            self.left.get_subscription_count() == 0
            or self.right.get_subscription_count() == 0
            or (gripper_msg is not None and self.grippers.get_subscription_count() == 0)
        ):
            if time.monotonic() >= deadline:
                raise RuntimeError("Arm or gripper controller subscribers not found")
            rclpy.spin_once(self.node, timeout_sec=0.1)

        duration = float(stage["duration"])
        left_msg = make_trajectory("left", stage["left_positions"], duration)
        right_msg = make_trajectory("right", stage["right_positions"], duration)
        for _ in range(3):
            self.left.publish(left_msg)
            self.right.publish(right_msg)
            if gripper_msg is not None:
                gripper_msg.header.stamp = self.node.get_clock().now().to_msg()
                self.grippers.publish(gripper_msg)
            rclpy.spin_once(self.node, timeout_sec=0.02)

        self.node.get_logger().info(f"Plan sent; waiting {duration:g} s before next stage")
        end = time.monotonic() + duration
        next_gripper_publish = time.monotonic() + 0.1
        prefetched = False
        while time.monotonic() < end:
            if gripper_msg is not None and time.monotonic() >= next_gripper_publish:
                gripper_msg.header.stamp = self.node.get_clock().now().to_msg()
                self.grippers.publish(gripper_msg)
                next_gripper_publish += 0.1
            if prefetch is not None and not prefetched and end - time.monotonic() <= prefetch_before:
                prefetch()
                prefetched = True
            remaining = max(0.0, end - time.monotonic())
            rclpy.spin_once(self.node, timeout_sec=min(0.1, remaining))
        if prefetch is not None and not prefetched:
            prefetch()

    def close(self) -> None:
        self.node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run planned poses followed by VLA control")
    parser.add_argument("config", type=Path, nargs="?", default=Path(__file__).with_name("stages.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        stages = json.load(file)["stages"]

    planner = None
    vla_process = None
    gate_dir = None
    ready_file = None
    inference_file = None
    execute_file = None

    def start_vla(stage: dict) -> None:
        nonlocal vla_process, gate_dir, ready_file, inference_file, execute_file
        if vla_process is not None:
            return
        gate_dir = Path(tempfile.mkdtemp(prefix="openpi-vla-gate-"))
        ready_file = gate_dir / "ready"
        inference_file = gate_dir / "infer"
        execute_file = gate_dir / "execute"
        print("Starting VLA standby process...", flush=True)
        vla_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robot_client.main",
                "--config",
                str(stage["config"]),
                "--skip-home",
                "--start-signal-file",
                str(execute_file),
                "--inference-signal-file",
                str(inference_file),
                "--ready-signal-file",
                str(ready_file),
            ]
        )

    def wait_for_vla_ready() -> None:
        print("Waiting for VLA standby readiness...", flush=True)
        while not ready_file.exists():
            returncode = vla_process.poll()
            if returncode is not None:
                raise subprocess.CalledProcessError(returncode, vla_process.args)
            time.sleep(0.05)
        print("VLA standby ready; starting plan stages.", flush=True)

    def request_vla_inference() -> None:
        if not inference_file.exists():
            inference_file.touch()
            print("VLA initial inference requested.", flush=True)

    try:
        vla_stages = [stage for stage in stages if stage["mode"] == "vla"]
        if len(vla_stages) != 1 or stages[-1]["mode"] != "vla":
            raise ValueError("exactly one vla stage is required and it must be last")
        start_vla(vla_stages[0])
        wait_for_vla_ready()

        for index, stage in enumerate(stages, start=1):
            mode = stage["mode"]
            print(f"Stage {index}/{len(stages)}: {mode}", flush=True)
            if mode == "plan":
                planner = planner or Planner()
                next_stage = stages[index] if index < len(stages) else None
                if next_stage is not None and next_stage["mode"] == "vla":
                    lead = float(next_stage.get("prefetch_before", 1.0))
                    if lead < 0:
                        raise ValueError("prefetch_before must be >= 0")
                    planner.move(stage, request_vla_inference, lead)
                else:
                    planner.move(stage)
            elif mode == "vla":
                if index != len(stages):
                    raise ValueError("vla must be the last stage")
                if planner is not None:
                    planner.close()
                    planner = None
                request_vla_inference()
                execute_file.touch()
                print("Plan complete; VLA start signal sent.", flush=True)
                if vla_process.wait() != 0:
                    raise subprocess.CalledProcessError(vla_process.returncode, vla_process.args)
            else:
                raise ValueError(f"Unknown control mode: {mode!r}")
    finally:
        if planner is not None:
            planner.close()
        if vla_process is not None and vla_process.poll() is None:
            vla_process.terminate()
            vla_process.wait(timeout=5)
        for signal_file in (ready_file, inference_file, execute_file):
            if signal_file is not None:
                signal_file.unlink(missing_ok=True)
        if gate_dir is not None:
            gate_dir.rmdir()


if __name__ == "__main__":
    main()
