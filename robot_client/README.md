# NZ100 Robot Client

这个目录用于部署在机器人电脑上，通过 OpenPI 的 `openpi-client` 连接 GPU 电脑上的 policy server。

当前支持普通同步推理和两种 RTC 推理：

```text
机器人电脑采集 top image + left/right joints + left/right gripper
→ WebSocket 发给 GPU server
→ 接收 action chunk
→ 机器人电脑执行动作
```

GPU server 默认地址：

```text
172.22.1.127:8000
```

## 安装

机器人电脑只需要安装轻量 client：

```bash
pip install -e /path/to/openpi/packages/openpi-client
```

如果 `robot_client` 也拷贝到机器人电脑，可以直接在 OpenPI 仓库根目录运行。

## GPU 电脑启动 server

在 GPU 电脑上先启动：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_nz100 \
  --policy.dir=/path/to/checkpoint_step \
  --port=8000
```

## 测试连接

在机器人电脑上可以先跑 mock 数据测试：

```bash
python -m robot_client.main --mock --once --prompt "pick up the bottle"
```

这只会发送随机图像和零关节，不会控制真实机器人。

## 接入真实机器人

真实 ROS2 话题配置在：

```text
robot_client/configs/nz100_client.yaml
```

默认话题来自 `lerobot_data_collection` 和 `robot_control_pc`：

| 功能 | 默认话题 | 消息类型 |
| --- | --- | --- |
| 顶部相机 | `/top/image_raw` | `sensor_msgs/msg/Image` |
| 关节状态 | `/joint_states` | `sensor_msgs/msg/JointState` |
| 左臂控制 | `/arm_left_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` |
| 右臂控制 | `/arm_right_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` |
| 双夹爪状态 | `/robot/api/io/state` | `interfaces/msg/Modbus` |
| 双夹爪控制 | `/robot/api/io/cmd` | `interfaces/msg/Modbus` |

当前 NZ100 action 输出为 16 维：

```text
0:7    左臂 7 个关节
7      左夹爪，PLC语义 1=开，2=关
8:15   右臂 7 个关节
15     右夹爪，PLC语义 1=开，2=关
```

state 发送给 OpenPI server 前会组织成原始 30 维布局：

```text
0:7     左臂关节位置
14:21   右臂关节位置
28      左夹爪，PLC语义 1=开，2=关
29      右夹爪，PLC语义 1=开，2=关
```

client 发给 OpenPI server 的 observation 结构是：

```python
{
    "images": {"cam_high": top_image},
    "state": state_30d,
    "prompt": language_instruction,
}
```

这会直接进入 server 端的 `NZ100Inputs` transform。

真实运行：

```bash
python -m robot_client.main --config robot_client/configs/nz100_client.yaml
```

只跑一次 action chunk：

```bash
python -m robot_client.main --config robot_client/configs/nz100_client.yaml --once
```

临时覆盖语言指令：

```bash
python -m robot_client.main \
  --config robot_client/configs/nz100_client.yaml \
  --prompt "pick up the bottle"
```

## 运行参数

```yaml
policy_host: 172.22.1.127
policy_port: 8000
control_fps: 100
open_loop_horizon: 32
max_steps: 0
point_time_from_start: 0.01
home_on_start: true
left_home_positions: [0.36, 0.36, -0.01, 1.92, 1.57, 0.00, -1.40]
right_home_positions: [-0.36, 0.36, -0.01, 1.92, 1.57, 0.00, 0.78]
home_time_from_start: 4.0
language_instruction: pick up the bottle and place it in the blue box
execution_mode: sync_chunk
rtc_execute_horizon: 8
rtc_prefix_len: 5
rtc_guidance_weight: 5.0
rtc_decay_tau: 3.0
```

含义：

| 参数 | 作用 |
| --- | --- |
| `policy_host` / `policy_port` | GPU policy server 地址 |
| `control_fps` | 机器人本地动作执行频率 |
| `open_loop_horizon` | 每次 action chunk 实际执行前多少步 |
| `max_steps` | 最多执行多少个动作步；`0` 表示一直跑 |
| `point_time_from_start` | 每个 JointTrajectoryPoint 的到达时间 |
| `home_on_start` | 是否在启动策略前让双臂回位并打开夹爪；默认 `true` |
| `left_home_positions` / `right_home_positions` | 左右臂启动回位的 7 关节目标位置 |
| `home_time_from_start` | 左右臂共用的启动回位轨迹时间；等待该时间后开始推理 |
| `language_instruction` | 发给模型的语言指令 |
| `execution_mode` | 推理方式选择：`sync_chunk` / `rtc_prefix` / `rtc_guidance` |
| `rtc_execute_horizon` | RTC 每次从当前 chunk 实际执行多少步 |
| `rtc_prefix_len` | RTC 使用上一段 chunk 前缀约束的步数 |
| `rtc_guidance_weight` | `rtc_guidance` 的引导强度 |
| `rtc_decay_tau` | `rtc_guidance` soft mask 衰减参数 |

可选 `execution_mode`：

```yaml
execution_mode: sync_chunk
execution_mode: rtc_prefix
execution_mode: rtc_guidance
```

`sync_chunk` 是普通 OpenPI chunk 推理；`rtc_prefix` 会硬锁上一段 action 前缀；`rtc_guidance` 会用 soft guidance 约束 chunk 连续性。默认配置仍是 `sync_chunk`，不启用 RTC 时不会影响普通推理。

启动和运行过程中，如果相机或关节状态还没到达，client 会一直等待对应 ROS2 topic 的第一帧数据。

如需启动时先回位再执行策略，在配置文件中设置：

```yaml
home_on_start: true
```

程序会同时发布左右臂回位轨迹和一次双夹爪打开命令。左右臂共用 4 秒到达时间，等待 4 秒后才进行第一次策略推理。设置为 `false` 可跳过启动回位和这次夹爪命令。

## 夹爪模式

默认使用双夹爪 Modbus：

```yaml
gripper_cmd_topic: /robot/api/io/cmd
left_gripper_key: an_out_d9746
right_gripper_key: an_out_d9747
modbus_open_value: 1
modbus_closed_value: 2
```

这和训练数据保持一致：client 发给模型的 state 夹爪值是 `1/2`，模型返回的 action 夹爪值也按 `1/2` 离散化后下发。

RTC 推理逻辑单独放在 `rtc_client.py`，普通同步逻辑仍在 `sync_client.py`，两条路径通过 `execution_mode` 选择。
