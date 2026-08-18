# Local M7.2 YOLO service

这个目录把原来的 M7.2 bottle target 服务封装成本地版：

```text
ROS2 RGB-D topic -> best_model.pt -> depth几何 -> /api/v1/target/latest
```

它保留 `multi_stage_control` 需要的 HTTP 接口：

```text
GET /health
GET /api/v1/target/latest?require_stable=0
```

启动：

```bash
cd /home/pc/VLA/openpi
source /opt/ros/humble/setup.bash
python3 -m multi_stage_control.yolo_detection.local_m72_service \
  --config multi_stage_control/yolo_detection/config.yaml \
  --robot
```

或者：

```bash
bash multi_stage_control/yolo_detection/run_local_m72_service.sh
```

`--robot` 会启用 YSRobot SDK 读取当前左手姿态，用于 `/api/v1/target/latest` 计算最终 `move_l` 的 flange target。

然后把 `multi_stage_control/stages.yaml` 里的 YOLO 地址改成本机服务：

```yaml
yolo_url: "http://127.0.0.1:19227/api/v1/target/latest?require_stable=0"
```

依赖：

```bash
python3 -m pip install --user \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
  
python3 -m pip install ultralytics pyyaml opencv-python
```

ROS2 相关依赖需要使用和 ROS2 Humble 匹配的 Python 3.10 环境。
