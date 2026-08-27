"""Lightweight command trajectory logger for NZ100 robot-client runs."""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from robot_client.config import ClientConfig
from robot_client.state_builder import NZ100Action


@dataclasses.dataclass(frozen=True)
class TrajectoryRecord:
    step: int
    monotonic_s: float
    elapsed_s: float
    left_joints: tuple[float, ...]
    left_gripper: float
    right_joints: tuple[float, ...]
    right_gripper: float


class TrajectoryLogger:
    """Collects commanded actions and writes CSV/JSON/PNG at shutdown."""

    def __init__(self, root_dir: str | Path, *, config: ClientConfig) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root_dir).expanduser() / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._records: list[TrajectoryRecord] = []
        self._start_monotonic = time.monotonic()
        self._closed = False

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "execution_mode": config.execution_mode,
            "control_hz": config.control_hz,
            "open_loop_horizon": config.open_loop_horizon,
            "max_steps": config.max_steps,
            "prompt": config.prompt,
        }
        (self.run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Trajectory logging enabled: {self.run_dir}")

    def record_action(self, action: NZ100Action) -> None:
        if self._closed:
            return
        now = time.monotonic()
        self._records.append(
            TrajectoryRecord(
                step=len(self._records),
                monotonic_s=now,
                elapsed_s=now - self._start_monotonic,
                left_joints=tuple(float(v) for v in np.asarray(action.left_joints, dtype=np.float32)),
                left_gripper=float(action.left_gripper),
                right_joints=tuple(float(v) for v in np.asarray(action.right_joints, dtype=np.float32)),
                right_gripper=float(action.right_gripper),
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._write_csv()
        self._write_json()
        self._write_plot()
        print(f"Trajectory log saved: {self.run_dir}")

    def _write_csv(self) -> None:
        csv_path = self.run_dir / "trajectory.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "step",
                    "elapsed_s",
                    *[f"left_j{i + 1}" for i in range(7)],
                    "left_gripper",
                    *[f"right_j{i + 1}" for i in range(7)],
                    "right_gripper",
                ]
            )
            for record in self._records:
                writer.writerow(
                    [
                        record.step,
                        f"{record.elapsed_s:.6f}",
                        *[f"{v:.9g}" for v in record.left_joints],
                        f"{record.left_gripper:.9g}",
                        *[f"{v:.9g}" for v in record.right_joints],
                        f"{record.right_gripper:.9g}",
                    ]
                )

    def _write_json(self) -> None:
        json_path = self.run_dir / "trajectory.json"
        payload = [dataclasses.asdict(record) for record in self._records]
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_plot(self) -> None:
        if not self._records:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Trajectory plot skipped: matplotlib unavailable ({exc})")
            return

        elapsed = np.asarray([record.elapsed_s for record in self._records], dtype=np.float32)
        left = np.asarray([record.left_joints for record in self._records], dtype=np.float32)
        right = np.asarray([record.right_joints for record in self._records], dtype=np.float32)
        grippers = np.asarray(
            [[record.left_gripper, record.right_gripper] for record in self._records],
            dtype=np.float32,
        )

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for idx in range(left.shape[1]):
            axes[0].plot(elapsed, left[:, idx], label=f"left_j{idx + 1}")
        axes[0].set_ylabel("left joints")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(ncol=4, fontsize=8)

        for idx in range(right.shape[1]):
            axes[1].plot(elapsed, right[:, idx], label=f"right_j{idx + 1}")
        axes[1].set_ylabel("right joints")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(ncol=4, fontsize=8)

        axes[2].step(elapsed, grippers[:, 0], where="post", label="left_gripper")
        axes[2].step(elapsed, grippers[:, 1], where="post", label="right_gripper")
        axes[2].set_ylabel("grippers")
        axes[2].set_xlabel("elapsed time (s)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(self.run_dir / "trajectory.png", dpi=150)
        plt.close(fig)
