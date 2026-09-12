# NZ100 独立轨迹回放

这个目录直接读取 LeRobot episode，不导入 `robot_client` 或 `python_sdk`。离线回放不需要 ROS2；真实执行只发布两路关节轨迹，不订阅相机或机器人状态。常用参数统一放在 `replay/config.yaml`。

默认只打印动作，不控制机器人：

先修改 `replay/config.yaml` 中的 `dataset`、`episode` 和其他参数，然后运行：

```bash
python3 -m replay.replay_episode
```

也可以使用另一个配置文件：

```bash
python3 -m replay.replay_episode --config /path/to/replay.yaml
```

显示该 episode 中的全部相机视角：

```bash
python3 -m replay.replay_episode \
  --dataset /path/to/data_lerobot \
  --episode 0 \
  --execute \
  --show-video
```

命令行参数会覆盖 YAML 中的同名设置，适合临时测试。

只显示指定视角：

```bash
python3 -m replay.replay_episode \
  --dataset /path/to/data_lerobot \
  --episode 0 \
  --show-video \
  --video-keys observation.images.top observation.images.wrist_left
```

按 `q` 可以停止。多视角在同一个窗口中自动平铺；不同分辨率会等比例缩放并补黑边。

真实执行时，关节与 `robot_client` 一样发布 ROS2 `JointTrajectory`，夹爪通过 YSRobot HTTP Modbus 控制：

```bash
python3 -m replay.replay_episode \
  --dataset /path/to/data_lerobot \
  --episode 0 \
  --execute \
  --robot-host 192.168.2.123 \
  --show-video
```

真实执行前需要加载 ROS2 环境：

```bash
source /opt/ros/humble/setup.bash
```

脚本默认先回位；使用 `--no-home` 跳过。`--execute` 会真实控制机器人，请先用默认 dry-run 检查动作和视角。

数据中的 `action` 必须是：

```text
0:7     左臂7个关节
7       左夹爪，1=开，2=关
8:15    右臂7个关节
15      右夹爪，1=开，2=关
```

`action` 可以包含更多维度。例如30维 TCP 数据集会使用前16维关节/夹爪动作，并忽略后14维左右 TCP。

依赖仅为 `numpy`、`pandas`、`pyarrow`；显示视频时额外需要 `opencv-python`。
