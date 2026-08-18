#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python3 -m multi_stage_control.yolo_detection.local_m72_service \
  --config multi_stage_control/yolo_detection/config.yaml \
  --robot
