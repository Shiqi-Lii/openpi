# Multi-stage VLA interruption control

这个目录现在用于运行一个常驻 VLA 编排器：

1. VLA 先启动并持续执行；
2. 监控线程实时获取 YOLO 瓶子中心 `xyz` 和左手 TCP `xyz`；
3. 两者距离小于阈值时，请求 VLA 在动作步边界暂停；
4. 执行中间抓取策略：`move_l -> GRASP`，确认返回后关左夹爪，再沿 robot/default base 的 `+Z` 方向 `move_l -> LIFT`；
5. 抓取策略结束后，VLA 重新读取当前 observation，重新请求 action chunk，并继续执行。

当前中间抓取策略位置在：

```text
multi_stage_control/run.py::_grasp_policy
```

## 运行

真实机器人：

```bash
python3 multi_stage_control/run.py
```

指定配置：

```bash
python3 multi_stage_control/run.py multi_stage_control/stages.yaml
```

Mock 检查流程：

```bash
python3 multi_stage_control/run.py --mock
```

## 配置

`multi_stage_control/stages.yaml` 仍然是 JSON 兼容写法：

```json
{
  "robot_client_config": "robot_client/configs/nz100_client.yaml",
  "yolo_url": "http://192.168.2.201:19227/api/v1/target/latest?require_stable=0",
  "yolo_timeout_s": 1.0,
  "poll_hz": 20.0,
  "distance_threshold_m": 0.05,
  "cooldown_s": 2.0,
  "max_interruptions": 1,
  "left_pose_frame": null,
  "grasp_move_vel": 5.0,
  "grasp_move_acc": 20.0,
  "grasp_move_planner": "pilz",
  "gripper_register": 9661,
  "gripper_close_value": 2,
  "gripper_settle_s": 1.0,
  "lift_mm": 60.0,
  "no_lift": false,
  "mock_yolo_xyz": null
}
```

字段说明：

- `robot_client_config`：VLA 客户端配置文件。
- `yolo_url`：YOLO HTTP 服务地址，默认读取 `target.pose_7d_m`。
- `yolo_timeout_s`：YOLO HTTP 请求超时。
- `poll_hz`：距离检测频率。
- `distance_threshold_m`：瓶子中心和左手 TCP 距离小于该值时触发暂停。
- `cooldown_s`：触发后的冷却时间，避免重复触发。
- `max_interruptions`：最多触发几次；`0` 表示不限制。
- `left_pose_frame`：传给 YSRobot SDK `get_pose(..., frame=...)` 的坐标系，`null` 使用 SDK 默认。
- `grasp_move_vel` / `grasp_move_acc` / `grasp_move_planner`：中间抓取策略调用 `move_l` 的参数。
- `handoff_hold_enabled` / `handoff_hold_s`：VLA 暂停后、第一次 `move_l` 前是否发布一次当前关节位置保持指令，以及保持时间；用于减小控制器交接抽动。
- `gripper_register` / `gripper_close_value` / `gripper_settle_s`：左夹爪关闭使用的 Modbus 参数，默认寄存器 `9661`、关闭值 `2`。
- `lift_mm`：抓取后沿 robot/default base `+Z` 方向提升的距离，默认 `60mm`。
- `no_lift`：为 `true` 时，只抓取并关夹爪，不执行 lift。
- `mock_yolo_xyz`：调试用固定瓶子坐标；真实服务接好后设为 `null`。

YOLO 服务当前按如下格式解析：

```json
{
  "valid": true,
  "target": {
    "pose_7d_m": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
  }
}
```

当 `valid=false` 时，本次检测不会触发中断。

中间策略执行顺序：

```text
[1/3] move_l -> GRASP
[2/3] CLOSE gripper: write_modbus(9661, 2)
[3/3] move_l -> LIFT
```

LIFT 位姿计算方式：

```text
lift_xyz = [target_x, target_y, target_z + lift_mm / 1000]
lift_quat = target_quat
```

三步完成后自动恢复 VLA；恢复时 VLA 会重新读取当前 observation 并请求新的 action chunk。

## 重要行为

暂停不是通过文件触发，而是进程内 `threading.Event`。VLA 客户端、ROS2 订阅、WebSocket
连接都不会关闭。

恢复时不会继续使用暂停前剩余的旧 action chunk，而是重新读取当前相机和机器人状态，再请求新的 action chunk。这个设计是为了避免抓取策略改变场景后，VLA 继续执行过期动作。
