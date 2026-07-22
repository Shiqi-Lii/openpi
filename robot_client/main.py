"""Entry point for NZ100 robot-side OpenPI policy inference.

The mock mode is safe for checking network/model connectivity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_client.config import ClientConfig
from robot_client.config import load_app_config
from robot_client.ros2_io import NZ100Ros2IO
from robot_client.runners import async_queue
from robot_client.runners import rtc_guidance
from robot_client.runners import sync_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NZ100 normal synchronous OpenPI client")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("robot_client/configs/nz100_client.yaml"),
        help="YAML config for server and ROS2 topics",
    )
    parser.add_argument("--host", default=None, help="Override GPU policy server IP/hostname")
    parser.add_argument("--port", type=int, default=None, help="Override GPU policy server port")
    parser.add_argument("--prompt", default=None, help="Override language instruction")
    parser.add_argument("--control-hz", type=float, default=None, help="Override local action execution rate")
    parser.add_argument(
        "--execution-mode",
        choices=("sync_chunk", "async_queue", "rtc_guidance"),
        default=None,
        help="Override inference/execution mode",
    )
    parser.add_argument("--mock", action="store_true", help="Use fake camera/state and print actions only")
    parser.add_argument("--once", action="store_true", help="Run one inference request and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = load_app_config(args.config)

    client_base = app_config.client
    config = ClientConfig(
        server_host=client_base.server_host if args.host is None else args.host,
        server_port=client_base.server_port if args.port is None else args.port,
        prompt=client_base.prompt if args.prompt is None else args.prompt,
        image_size=client_base.image_size,
        control_hz=client_base.control_hz if args.control_hz is None else args.control_hz,
        open_loop_horizon=client_base.open_loop_horizon,
        max_steps=client_base.max_steps,
        action_refill_threshold=client_base.action_refill_threshold,
        execution_mode=client_base.execution_mode if args.execution_mode is None else args.execution_mode,
        execute_full_chunk=client_base.execute_full_chunk,
        rtc_execute_horizon=client_base.rtc_execute_horizon,
        rtc_prefix_len=client_base.rtc_prefix_len,
        rtc_guidance_weight=client_base.rtc_guidance_weight,
        rtc_decay_tau=client_base.rtc_decay_tau,
        rtc_decay_end=client_base.rtc_decay_end,
        rtc_use_vjp=client_base.rtc_use_vjp,
        rtc_delay_buffer_size=client_base.rtc_delay_buffer_size,
        rtc_max_delay_steps=client_base.rtc_max_delay_steps,
    )

    print(
        "Starting NZ100 OpenPI client: "
        f"server=tcp://{config.server_host}:{config.server_port}, "
        f"mode={config.execution_mode}, "
        f"control_hz={config.control_hz}, "
        f"open_loop_horizon={config.open_loop_horizon}, "
        f"action_refill_threshold={config.action_refill_threshold}, "
        f"max_steps={config.max_steps}, "
        f"mock={args.mock}"
    )
    print(f"Language instruction: {config.prompt!r}")

    ros_io = None
    try:
        if not args.mock:
            ros_io = NZ100Ros2IO(app_config.ros2)
            ros_io.connect()
            if app_config.ros2.home_on_start:
                ros_io.move_to_home()
            else:
                print("Skipping NZ100 startup pose command.")

        if config.execution_mode == "sync_chunk":
            sync_chunk.run(config, ros_io=ros_io, mock=args.mock, once=args.once)
        elif config.execution_mode == "async_queue":
            async_queue.run(config, ros_io=ros_io, mock=args.mock, once=args.once)
        elif config.execution_mode == "rtc_guidance":
            rtc_guidance.run(config, ros_io=ros_io, mock=args.mock, once=args.once)
        else:
            raise ValueError(f"Unsupported execution_mode: {config.execution_mode!r}")
    finally:
        if ros_io is not None:
            ros_io.disconnect()


if __name__ == "__main__":
    main()
