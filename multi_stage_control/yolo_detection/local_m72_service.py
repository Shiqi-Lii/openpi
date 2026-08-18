#!/usr/bin/env python3
"""Local M7.2-style bottle target service.

This keeps the original HTTP interface used by ``multi_stage_control`` while
replacing the old board/relay/RKNN pipeline with:

  ROS2 RGB-D topics -> Ultralytics YOLO .pt -> depth geometry -> HTTP latest.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"


def mono_ms() -> float:
    return time.monotonic() * 1000.0


def wall_ms() -> int:
    return int(time.time() * 1000)


def elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def qnormalize(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise RuntimeError("zero quaternion")
    return q / n


def q_to_R(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = qnormalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


class LatestSlot:
    def __init__(self) -> None:
        self.value = None
        self.seq = 0
        self.overwrites = 0
        self.cv = threading.Condition()

    def put(self, value) -> None:
        with self.cv:
            if self.value is not None:
                self.overwrites += 1
            self.value = value
            self.seq += 1
            self.cv.notify_all()

    def take_new(self, last_seq: int, timeout: float = 1.0):
        deadline = time.monotonic() + timeout
        with self.cv:
            while self.seq <= last_seq:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return last_seq, None
                self.cv.wait(remain)
            value = self.value
            self.value = None
            return self.seq, value


class Ros2RgbdSource:
    """Latest-only ROS2 RGB-D source."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._rclpy = None
        self._executor = None
        self._node = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._color_msg = None
        self._depth_msg = None
        self._camera_info_msg = None
        self._pair_id = 0

    def start(self) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()

        class _Node(Node):
            pass

        ros_cfg = self.cfg["ros2"]
        self._node = _Node("local_m72_yolo_detection")
        self._node.create_subscription(Image, str(ros_cfg["color_topic"]), self._on_color, 10)
        self._node.create_subscription(Image, str(ros_cfg["depth_topic"]), self._on_depth, 10)
        self._node.create_subscription(CameraInfo, str(ros_cfg["camera_info_topic"]), self._on_camera_info, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, name="local_m72_ros2", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()

    def _spin(self) -> None:
        while self._rclpy is not None and self._rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:
                break

    def _on_color(self, msg) -> None:
        with self._lock:
            self._color_msg = msg

    def _on_depth(self, msg) -> None:
        with self._lock:
            self._depth_msg = msg

    def _on_camera_info(self, msg) -> None:
        with self._lock:
            self._camera_info_msg = msg

    def latest(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        with self._lock:
            color_msg = self._color_msg
            depth_msg = self._depth_msg
            info_msg = self._camera_info_msg
        if color_msg is None or depth_msg is None:
            raise RuntimeError("WAIT_ROS2_RGBD")

        rgb = image_msg_to_rgb(color_msg)
        depth = image_msg_to_depth_u16(depth_msg)
        if info_msg is not None:
            k = list(info_msg.k)
            self.cfg["camera"]["fx"] = float(k[0])
            self.cfg["camera"]["fy"] = float(k[4])
            self.cfg["camera"]["cx"] = float(k[2])
            self.cfg["camera"]["cy"] = float(k[5])
            self.cfg["camera"]["width"] = int(info_msg.width)
            self.cfg["camera"]["height"] = int(info_msg.height)
        self._pair_id += 1
        return {
            "pair_id": self._pair_id,
            "rgb": rgb,
            "depth": depth,
            "meta": {
                "color": {"width": int(color_msg.width), "height": int(color_msg.height), "encoding": color_msg.encoding},
                "depth": {"width": int(depth_msg.width), "height": int(depth_msg.height), "encoding": depth_msg.encoding},
            },
            "capture_wall_ms": wall_ms(),
            "capture_received_mono_ms": mono_ms(),
            "ros2_capture_ms": elapsed_ms(t0),
        }


class YoloPtRuntime:
    def __init__(self, cfg: dict[str, Any]) -> None:
        from ultralytics import YOLO

        ycfg = cfg["yolo"]
        model_path = Path(str(ycfg["model_path"]))
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        self.model = YOLO(str(model_path))
        self.cfg = cfg

    def infer(self, rgb: np.ndarray) -> tuple[dict[str, Any], float]:
        ycfg = self.cfg["yolo"]
        t0 = time.perf_counter()
        kwargs = {
            "conf": float(ycfg["conf"]),
            "imgsz": int(ycfg["imgsz"]),
            "verbose": False,
        }
        if ycfg.get("device") is not None:
            kwargs["device"] = ycfg["device"]
        results = self.model.predict(rgb, **kwargs)
        infer_ms = elapsed_ms(t0)
        detections = yolo_results_to_detections(results[0], self.cfg)
        return {"status": "ok", "detections": detections, "timing": {"total_ms": infer_ms}}, infer_ms


class RobotPoseReader:
    def __init__(self, cfg: dict[str, Any], enabled: bool) -> None:
        self.cfg = cfg
        self.enabled = enabled
        self.robot = None
        self.ArmType = None
        self.lock = threading.Lock()
        if enabled:
            self.connect()

    def connect(self) -> None:
        rc = self.cfg["robot"]
        sdk_path = rc.get("sdk_path") or os.environ.get("VISIONOPS_ROBOT_SDK_PATH")
        if sdk_path and sdk_path not in sys.path:
            sys.path.insert(0, str(sdk_path))
        from ysrobot import ArmType, RobotClient

        robot = RobotClient(str(rc["host"]), port=int(rc["port"]), timeout_ms=int(rc["timeout_ms"]))
        ret = robot.login(str(rc["login_level"]), str(rc["login_pin"]))
        if not ret.success:
            raise RuntimeError(f"robot login failed: {ret.message}")
        ret = robot.connect()
        if not ret.success:
            raise RuntimeError(f"robot connect failed: {ret.message}")
        self.robot = robot
        self.ArmType = ArmType

    def pose_mm(self) -> list[float]:
        if not self.enabled:
            raise RuntimeError("robot pose reader disabled")
        with self.lock:
            try:
                p = self.robot.motion.get_pose(self.ArmType.Left)
            except Exception:
                try:
                    self.robot.disconnect()
                except Exception:
                    pass
                self.connect()
                p = self.robot.motion.get_pose(self.ArmType.Left)
        return [float(p.x) * 1000.0, float(p.y) * 1000.0, float(p.z) * 1000.0, float(p.qx), float(p.qy), float(p.qz), float(p.qw)]


class TablePlane:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        tc = cfg["table_plane"]
        self.n = np.asarray(tc["seed_normal_camera"], float)
        self.n /= np.linalg.norm(self.n)
        self.d = float(tc["seed_d_m"])
        self.source = "seed"
        self.path = Path(str(tc["cache_path"]))
        self.rng = np.random.default_rng(7202)
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                n = np.asarray(d["normal_camera"], float)
                self.n = n / np.linalg.norm(n)
                self.d = float(d["d_m"])
                self.source = "disk_cache"
            except Exception:
                pass

    def _roi(self, bbox, w: int, h: int):
        tc = self.cfg["table_plane"]
        x1, y1, x2, y2 = map(float, bbox)
        return (
            max(0, int(x1 - float(tc["roi_expand_x_px"]))),
            max(0, int(y1 - float(tc["roi_expand_y_up_px"]))),
            min(w, int(x2 + float(tc["roi_expand_x_px"]))),
            min(h, int(y2 + float(tc["roi_expand_y_down_px"]))),
        )

    def validate_or_refit(self, depth, bottle_mask, bottle_points, bbox):
        t0 = time.perf_counter()
        h, w = depth.shape
        tc = self.cfg["table_plane"]
        roi = self._roi(bbox, w, h)
        k = max(1, int(tc["exclude_mask_dilate_px"]))
        excl = cv2.dilate(bottle_mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        pts = depth_roi_points(depth, roi, excl, self.cfg, int(tc["validate_sample_count"]), self.rng)
        residual = np.abs(pts @ self.n + self.d) if len(pts) else np.empty(0)
        thr = float(tc["validate_inlier_threshold_mm"]) / 1000.0
        ratio = float(np.mean(residual <= thr)) if len(residual) else 0.0
        info = {"source": self.source, "validate_points": int(len(pts)), "validate_inlier_ratio": ratio, "refit": False}
        if ratio < float(tc["validate_minimum_inlier_ratio"]):
            pts = depth_roi_points(depth, roi, excl, self.cfg, int(tc["refit_sample_count"]), self.rng)
            if len(pts) < 200:
                raise RuntimeError("TABLE_REFIT_TOO_FEW_POINTS")
            best, best_count = None, -1
            rthr = float(tc["ransac_threshold_mm"]) / 1000.0
            for _ in range(int(tc["ransac_iterations"])):
                a, b, c = pts[self.rng.choice(len(pts), 3, replace=False)]
                n = np.cross(b - a, c - a)
                norm = float(np.linalg.norm(n))
                if norm < 1e-8:
                    continue
                n /= norm
                d = -float(np.dot(n, a))
                cnt = int(np.count_nonzero(np.abs(pts @ n + d) <= rthr))
                if cnt > best_count:
                    best, best_count = (n, d), cnt
            if best is None:
                raise RuntimeError("TABLE_REFIT_FAILED")
            n, d = best
            inl = np.abs(pts @ n + d) <= rthr
            P = pts[inl]
            if float(np.mean(inl)) < float(tc["refit_minimum_inlier_ratio"]):
                raise RuntimeError("TABLE_REFIT_LOW_INLIER")
            cen = P.mean(axis=0)
            _, _, vh = np.linalg.svd(P - cen, full_matrices=False)
            n = vh[-1]
            n /= np.linalg.norm(n)
            d = -float(np.dot(n, cen))
            if float(np.median(bottle_points @ n + d)) < 0:
                n, d = -n, -d
            self.n, self.d, self.source = n, d, "ransac_refit"
            info.update({"source": self.source, "refit": True, "refit_inlier_ratio": float(np.mean(inl))})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"normal_camera": n.tolist(), "d_m": d, "updated_ms": wall_ms(), "inlier_ratio": float(np.mean(inl))}, indent=2), encoding="utf-8")
        n, d = self.n.copy(), float(self.d)
        if float(np.median(bottle_points @ n + d)) < 0:
            n, d = -n, -d
        return n, d, roi, info, elapsed_ms(t0)


class ContinuousEngine:
    def __init__(self, cfg: dict[str, Any], robot_enabled: bool) -> None:
        self.cfg = cfg
        self.source = Ros2RgbdSource(cfg)
        self.runtime = YoloPtRuntime(cfg)
        self.robot = RobotPoseReader(cfg, robot_enabled)
        self.plane = TablePlane(cfg)
        self.capture_slot = LatestSlot()
        self.infer_slot = LatestSlot()
        self.stop_event = threading.Event()
        self.latest_lock = threading.Lock()
        self.latest_result = None
        self.latest_seq = 0
        self.result_history = collections.deque(maxlen=int(cfg["continuous"]["stability_window"]))
        self.errors = collections.deque(maxlen=20)
        self.counters = collections.Counter()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        self.source.start()
        for name, fn in [("capture", self.capture_loop), ("infer", self.infer_loop), ("geometry", self.geometry_loop)]:
            th = threading.Thread(target=fn, name=f"local-m72-{name}", daemon=True)
            th.start()
            self.threads.append(th)

    def stop(self) -> None:
        self.stop_event.set()
        self.source.close()

    def record_error(self, stage: str, e: Exception) -> None:
        self.counters[f"{stage}_errors"] += 1
        self.errors.append({"stage": stage, "error": str(e), "type": type(e).__name__, "wall_ms": wall_ms()})

    def capture_loop(self) -> None:
        hz = max(1.0, float(self.cfg["ros2"]["capture_hz"]))
        period = 1.0 / hz
        while not self.stop_event.is_set():
            t_cycle = time.monotonic()
            try:
                item = self.source.latest()
                self.capture_slot.put(item)
                self.counters["captured"] += 1
            except Exception as e:
                self.record_error("capture", e)
                time.sleep(0.05)
            remain = period - (time.monotonic() - t_cycle)
            if remain > 0:
                self.stop_event.wait(remain)

    def infer_loop(self) -> None:
        seq = 0
        while not self.stop_event.is_set():
            seq, item = self.capture_slot.take_new(seq, timeout=0.5)
            if item is None:
                continue
            try:
                result, infer_ms = self.runtime.infer(item["rgb"])
                det = select_bottle(result, self.cfg)
                self.infer_slot.put({
                    "pair_id": item["pair_id"],
                    "meta": item["meta"],
                    "depth": item["depth"],
                    "det": det,
                    "runtime": result,
                    "capture_received_mono_ms": item["capture_received_mono_ms"],
                    "ros2_capture_ms": item["ros2_capture_ms"],
                    "runtime_internal_ms": float((result.get("timing") or {}).get("total_ms") or infer_ms),
                    "infer_stage_ms": infer_ms,
                    "infer_done_mono_ms": mono_ms(),
                })
                self.counters["inferred"] += 1
            except Exception as e:
                self.record_error("infer", e)

    def geometry_loop(self) -> None:
        seq = 0
        T = np.asarray(self.cfg["hand_eye"]["T_base_camera_m"], float)
        while not self.stop_event.is_set():
            seq, item = self.infer_slot.take_new(seq, timeout=0.5)
            if item is None:
                continue
            t_stage = time.perf_counter()
            try:
                depth = np.asarray(item["depth"], dtype=np.uint16)
                h, w = depth.shape
                det = item["det"]
                t = time.perf_counter()
                mask = detection_mask(det, h, w)
                k = int(self.cfg["bottle"]["mask_erode_px"])
                eroded = cv2.erode(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0 if k > 0 else mask
                pts = depth_points(depth, eroded, self.cfg)
                mask_depth_ms = elapsed_ms(t)
                bbox = det.get("bbox_xyxy")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    raise RuntimeError("missing bbox_xyxy")
                n, d, _roi, plane_info, plane_ms = self.plane.validate_or_refit(depth, mask, pts, bbox)
                center_cam, fit, center_ms = fit_fixed_radius(pts, n, d, self.cfg)
                t = time.perf_counter()
                hp = T @ np.array([*center_cam, 1.0])
                center_base_mm = hp[:3] * 1000.0
                nbase = T[:3, :3] @ n
                nbase /= np.linalg.norm(nbase)
                transform_ms = elapsed_ms(t)
                result = {
                    "schema": "visionops.bottle_m7_2_grasp.local.v1",
                    "pair_id": int(item["pair_id"]),
                    "created_wall_ms": wall_ms(),
                    "frame": "robot_default_base",
                    "score": float(det.get("score", 0)),
                    "bbox_xyxy": [float(x) for x in bbox],
                    "grasp_center_camera_m": center_cam.tolist(),
                    "grasp_center_base_mm": center_base_mm.tolist(),
                    "table_normal_base": nbase.tolist(),
                    "quality": {
                        "valid": True,
                        "fit_rms_mm": float(fit["rms_residual_mm"]),
                        "fit_p95_mm": float(fit["p95_abs_residual_mm"]),
                        "table_inlier_ratio": float(plane_info.get("validate_inlier_ratio", 0.0)),
                        "table_source": plane_info.get("source"),
                    },
                    "timing_ms": {
                        "ros2_capture_ms": float(item["ros2_capture_ms"]),
                        "runtime_internal_ms": float(item["runtime_internal_ms"]),
                        "infer_stage_ms": float(item["infer_stage_ms"]),
                        "mask_depth_ms": mask_depth_ms,
                        "table_plane_ms": plane_ms,
                        "center_fit_ms": center_ms,
                        "transform_ms": transform_ms,
                        "geometry_stage_ms": elapsed_ms(t_stage),
                        "capture_to_result_ms": mono_ms() - float(item["capture_received_mono_ms"]),
                    },
                }
                self.publish_result(result)
                self.counters["geometry_ok"] += 1
            except Exception as e:
                self.record_error("geometry", e)

    def publish_result(self, result: dict[str, Any]) -> None:
        with self.latest_lock:
            self.result_history.append(result)
            window_n = int(self.cfg["continuous"]["stability_window"])
            recent = list(self.result_history)[-window_n:]
            pts = [np.asarray(r["grasp_center_base_mm"], float) for r in recent]
            if len(pts) >= window_n:
                steps = [float(np.linalg.norm(pts[i] - pts[i - 1])) for i in range(1, len(pts))]
                span = max(float(np.linalg.norm(pts[i] - pts[j])) for i in range(len(pts)) for j in range(i + 1, len(pts)))
                stable = max(steps or [0.0]) <= float(self.cfg["continuous"]["stable_max_step_mm"]) and span <= float(self.cfg["continuous"]["stable_max_span_mm"])
            else:
                steps, span, stable = [], None, False
            result["stability"] = {"window": len(pts), "required_window": window_n, "stable": bool(stable), "max_step_mm": None if not steps else max(steps), "span_mm": span}
            self.latest_result = result
            self.latest_seq += 1
            result["result_seq"] = self.latest_seq

    def snapshot(self):
        with self.latest_lock:
            return None if self.latest_result is None else json.loads(json.dumps(self.latest_result))

    def grasp_response(self, require_stable=None) -> dict[str, Any]:
        r = self.snapshot()
        if r is None:
            return {"status": "ok", "valid": False, "reason": "WAIT_FIRST_RESULT"}
        age_ms = max(0, wall_ms() - int(r["created_wall_ms"]))
        stable = bool(r["stability"]["stable"])
        req = bool(self.cfg["continuous"]["require_stable_default"]) if require_stable is None else bool(require_stable)
        valid = age_ms <= float(self.cfg["continuous"]["max_result_age_ms"]) and bool(r["quality"]["valid"]) and (stable or not req)
        if age_ms > float(self.cfg["continuous"]["max_result_age_ms"]):
            reason = "STALE_RESULT"
        elif req and not stable:
            reason = "TARGET_NOT_STABLE"
        elif not r["quality"]["valid"]:
            reason = "LOW_QUALITY"
        else:
            reason = "OK"
        return {"status": "ok", "valid": bool(valid), "reason": reason, "age_ms": age_ms, "result": r}

    def target_response(self, require_stable=None) -> dict[str, Any]:
        g = self.grasp_response(require_stable=require_stable)
        if not g["valid"]:
            return g
        if not self.robot.enabled:
            return {**g, "valid": False, "reason": "ROBOT_POSE_DISABLED"}
        t0 = time.perf_counter()
        pose = self.robot.pose_mm()
        robot_pose_ms = elapsed_ms(t0)
        q = qnormalize(pose[3:7])
        tool = np.asarray(self.cfg["tool"]["flange_to_grasp_local_mm"], float)
        pg = np.asarray(g["result"]["grasp_center_base_mm"], float)
        tool_base = q_to_R(q) @ tool
        target = pg - tool_base
        pose7_mm = [*target.tolist(), *q.tolist()]
        pose7_m = [target[0] / 1000.0, target[1] / 1000.0, target[2] / 1000.0, *q.tolist()]
        return {
            "status": "ok",
            "valid": True,
            "reason": "OK",
            "age_ms": g["age_ms"],
            "vision": {
                "pair_id": g["result"]["pair_id"],
                "result_seq": g["result"]["result_seq"],
                "score": g["result"]["score"],
                "stable": g["result"]["stability"]["stable"],
                "stability": g["result"]["stability"],
                "grasp_center_base_mm": g["result"]["grasp_center_base_mm"],
                "quality": g["result"]["quality"],
            },
            "current_left_flange_pose_mm": pose,
            "tool_offset_local_mm": tool.tolist(),
            "tool_offset_base_mm": tool_base.tolist(),
            "target": {
                "frame": "robot_default_base",
                "pose_7d_mm": pose7_mm,
                "pose_7d_m": pose7_m,
                "position_mm": target.tolist(),
                "orientation_xyzw": q.tolist(),
                "quaternion_policy": "COPY_CURRENT_VLA_LEFT_QUATERNION_UNCHANGED",
            },
            "target_calc_ms": robot_pose_ms,
        }

    def status(self) -> dict[str, Any]:
        latest = self.grasp_response(require_stable=False)
        return {
            "status": "ok",
            "runtime": "ultralytics_yolo_pt",
            "robot_pose_enabled": self.robot.enabled,
            "threads": {t.name: t.is_alive() for t in self.threads},
            "capture_queue_overwrites": self.capture_slot.overwrites,
            "infer_queue_overwrites": self.infer_slot.overwrites,
            "counters": dict(self.counters),
            "latest": {
                "valid": latest.get("valid"),
                "reason": latest.get("reason"),
                "age_ms": latest.get("age_ms"),
                "pair_id": ((latest.get("result") or {}).get("pair_id")),
                "result_seq": ((latest.get("result") or {}).get("result_seq")),
                "stable": (((latest.get("result") or {}).get("stability") or {}).get("stable")),
            },
            "recent_errors": list(self.errors)[-5:],
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "BottleM72Local/1.0"

    @property
    def engine(self) -> ContinuousEngine:
        return self.server.engine

    def log_message(self, fmt, *args) -> None:
        if self.path != "/health":
            print("[HTTP]", fmt % args)

    def json_response(self, code: int, obj: Any) -> None:
        raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    @staticmethod
    def bool_arg(qs, name: str, default: bool = False) -> bool:
        value = (qs.get(name) or [None])[0]
        if value is None:
            return default
        return str(value).lower() in {"1", "true", "yes", "on"}

    def do_GET(self) -> None:
        try:
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path == "/health":
                self.json_response(200, self.engine.status())
            elif u.path == "/api/v1/target/latest":
                self.json_response(200, self.engine.target_response(self.bool_arg(qs, "require_stable", False)))
            else:
                self.json_response(404, {"status": "error", "error": "not_found"})
        except BrokenPipeError:
            pass
        except Exception as e:
            self.json_response(500, {"status": "error", "error": str(e), "type": type(e).__name__})


def image_msg_to_rgb(msg) -> np.ndarray:
    height, width = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(enc)
    if channels is None:
        raise RuntimeError(f"unsupported color encoding: {msg.encoding}")
    row_bytes = width * channels
    rows = data[: height * int(msg.step)].reshape(height, int(msg.step))
    image = rows[:, :row_bytes].reshape(height, width, channels)
    if enc == "rgb8":
        return image.copy()
    if enc == "bgr8":
        return image[:, :, ::-1].copy()
    if enc == "mono8":
        return np.repeat(image, 3, axis=2)
    if enc == "rgba8":
        return image[:, :, :3].copy()
    return image[:, :, :3][:, :, ::-1].copy()


def image_msg_to_depth_u16(msg) -> np.ndarray:
    height, width = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()
    if enc in {"16uc1", "mono16"}:
        dtype = np.uint16
    elif enc == "32fc1":
        data = np.frombuffer(msg.data, dtype=np.float32)
        rows = data[: height * (int(msg.step) // 4)].reshape(height, int(msg.step) // 4)
        return np.nan_to_num(rows[:, :width] * 1000.0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16)
    else:
        raise RuntimeError(f"unsupported depth encoding: {msg.encoding}")
    data = np.frombuffer(msg.data, dtype=dtype)
    rows = data[: height * (int(msg.step) // np.dtype(dtype).itemsize)].reshape(height, int(msg.step) // np.dtype(dtype).itemsize)
    return rows[:, :width].copy()


def yolo_results_to_detections(result, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    names = getattr(result, "names", {}) or getattr(result, "names", {})
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    masks = result.masks
    detections = []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    polygons = None
    if masks is not None and getattr(masks, "xy", None) is not None:
        polygons = masks.xy
    for i in range(len(xyxy)):
        class_name = str(names.get(int(cls[i]), cls[i])) if isinstance(names, dict) else str(cls[i])
        det = {
            "class_name": class_name,
            "score": float(conf[i]),
            "bbox_xyxy": [float(x) for x in xyxy[i].tolist()],
        }
        if polygons is not None and i < len(polygons):
            det["mask"] = {"encoding": "polygon", "polygon": np.asarray(polygons[i], dtype=float).tolist()}
        detections.append(det)
    return detections


def select_bottle(runtime: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    rows = runtime.get("detections")
    if not isinstance(rows, list):
        raise RuntimeError("runtime response has no detections")
    name = str(cfg["yolo"]["class_name"])
    th = float(cfg["bottle"]["minimum_score"])
    rows = [r for r in rows if str(r.get("class_name")) == name and float(r.get("score", 0)) >= th]
    if not rows:
        raise RuntimeError("NO_BOTTLE")
    return max(rows, key=lambda r: float(r.get("score", 0)))


def detection_mask(det: dict[str, Any], h: int, w: int) -> np.ndarray:
    m = det.get("mask") or {}
    if m.get("encoding") == "polygon":
        poly = m.get("polygon")
        polys = [poly] if poly and isinstance(poly[0][0], (int, float)) else poly
        out = np.zeros((h, w), np.uint8)
        for p in polys:
            a = np.rint(np.asarray(p, float)).astype(np.int32).reshape(-1, 2)
            if len(a) >= 3:
                cv2.fillPoly(out, [a], 1)
        return out.astype(bool)
    x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
    out = np.zeros((h, w), bool)
    out[max(0, y1): min(h, y2), max(0, x1): min(w, x2)] = True
    return out


def depth_points(depth: np.ndarray, mask: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    cam = cfg["camera"]
    d = depth.astype(np.float64) * float(cam["depth_scale_m_per_unit"])
    valid = mask & (d >= float(cam["depth_min_mm"]) / 1000.0) & (d <= float(cam["depth_max_mm"]) / 1000.0)
    ys, xs = np.nonzero(valid)
    if len(xs) < 30:
        raise RuntimeError("TOO_FEW_DEPTH")
    z = d[ys, xs]
    x = (xs.astype(float) - float(cam["cx"])) / float(cam["fx"]) * z
    y = (ys.astype(float) - float(cam["cy"])) / float(cam["fy"]) * z
    return np.column_stack([x, y, z])


def depth_roi_points(depth, roi, exclude, cfg, max_points, rng) -> np.ndarray:
    x1, y1, x2, y2 = roi
    cam = cfg["camera"]
    d = depth[y1:y2, x1:x2].astype(np.float64) * float(cam["depth_scale_m_per_unit"])
    good = (d >= float(cam["depth_min_mm"]) / 1000.0) & (d <= float(cam["depth_max_mm"]) / 1000.0)
    if exclude is not None:
        good &= ~exclude[y1:y2, x1:x2]
    ys, xs = np.nonzero(good)
    if len(xs) > max_points:
        idx = rng.choice(len(xs), max_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    if not len(xs):
        return np.empty((0, 3))
    z = d[ys, xs]
    uu, vv = xs + x1, ys + y1
    x = (uu.astype(float) - float(cam["cx"])) / float(cam["fx"]) * z
    y = (vv.astype(float) - float(cam["cy"])) / float(cam["fy"]) * z
    return np.column_stack([x, y, z])


def fit_fixed_radius(points, n, d, cfg):
    t0 = time.perf_counter()
    bc = cfg["bottle"]
    heights = points @ n + d
    top = float(np.percentile(heights, 98))
    top_mm = top * 1000.0
    if not (float(bc["minimum_visible_height_mm"]) <= top_mm <= float(bc["maximum_visible_height_mm"])):
        raise RuntimeError(f"BAD_VISIBLE_HEIGHT:{top_mm:.1f}")
    lo = float(bc["fit_band_lo_fraction"]) * top
    hi = float(bc["fit_band_hi_fraction"]) * top
    band = points[(heights >= lo) & (heights <= hi)]
    if len(band) < int(bc["minimum_band_points"]):
        raise RuntimeError("TOO_FEW_BAND_POINTS")
    target = float(bc["target_height_fraction"]) * top
    centroid = np.median(points, axis=0)
    view = centroid - n * float(np.dot(centroid, n))
    view /= np.linalg.norm(view)
    transverse = np.cross(n, view)
    transverse /= np.linalg.norm(transverse)
    tt, qq = band @ transverse, band @ view
    radius = float(bc["nominal_diameter_mm"]) * 0.0005
    c = np.array([float(np.median(tt)), float(np.median(qq) + 0.75 * radius)])
    for _ in range(20):
        dx, dy = c[0] - tt, c[1] - qq
        dist = np.sqrt(dx * dx + dy * dy)
        good = dist > 1e-9
        r = dist[good] - radius
        J = np.column_stack([dx[good] / dist[good], dy[good] / dist[good]])
        med = float(np.median(np.abs(r))) + 1e-9
        hub = max(0.0015, 2.5 * med)
        w = np.minimum(1.0, hub / np.maximum(np.abs(r), 1e-12))
        A = J.T @ (w[:, None] * J)
        b = J.T @ (w * r)
        try:
            step = np.linalg.solve(A + np.eye(2) * 1e-10, b)
        except np.linalg.LinAlgError:
            break
        c -= step
        if float(np.linalg.norm(step)) < 1e-7:
            break
    rr = np.sqrt((tt - c[0]) ** 2 + (qq - c[1]) ** 2) - radius
    rms = float(np.sqrt(np.mean(rr * rr)) * 1000.0)
    p95 = float(np.percentile(np.abs(rr), 95) * 1000.0)
    if rms > float(bc["maximum_fit_rms_mm"]):
        raise RuntimeError(f"FIT_RMS_HIGH:{rms:.2f}")
    center = c[0] * transverse + c[1] * view + (target - d) * n
    return center, {"top_p98_mm": top_mm, "target_height_mm": target * 1000.0, "band_points": int(len(band)), "rms_residual_mm": rms, "p95_abs_residual_mm": p95}, elapsed_ms(t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--robot", action="store_true", help="enable /api/v1/target/latest by reading current left pose")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = ContinuousEngine(cfg, robot_enabled=args.robot)
    engine.start()
    host = args.host or str(cfg["service"]["host"])
    port = args.port or int(cfg["service"]["port"])
    server = ThreadingHTTPServer((host, port), Handler)
    server.engine = engine
    print("=" * 78)
    print("Local Bottle M7.2 YOLO Service")
    print(f"HTTP          : http://{host}:{port}")
    print(f"ROS2 color    : {cfg['ros2']['color_topic']}")
    print(f"ROS2 depth    : {cfg['ros2']['depth_topic']}")
    print(f"YOLO model    : {cfg['yolo']['model_path']}")
    print("Latest target : GET /api/v1/target/latest")
    print(f"Robot pose    : {'enabled' if args.robot else 'disabled'}")
    print("=" * 78)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.server_close()


if __name__ == "__main__":
    main()
