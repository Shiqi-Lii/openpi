# Multi-stage control

按 YAML 顺序执行双臂规划点，然后直接启动 VLA：

```bash
python3 multi_stage_control/run.py
```

也可以指定其他流程文件：

```bash
python3 multi_stage_control/run.py path/to/stages.yaml
```

`plan` 阶段的 `duration` 支持小数秒，左右关节位置各需要 7 个值。可以在 VLA
前配置多个 `plan` 阶段；`vla` 必须是最后一个阶段。VLA 启动时会自动跳过
`robot_client` 的 HOME 动作，直接从最后一个规划点接管控制。

VLA 进程会在任何规划命令发送前启动，提前完成 Python 导入、ROS2/DDS、传感器
等待和 WebSocket 连接，然后保持待命。VLA 阶段的 `prefetch_before` 只表示在前
一个规划阶段结束前多少秒请求首个动作 chunk。预取完成后 VLA 会等待执行闸门，
不会提前控制机械臂；规划结束并关闭规划 publisher 后才会开放闸门。建议该值略
大于正常推理延迟，例如推理约 `0.136s` 时可设置为 `0.2`。

每个 `plan` 还可使用 `left_gripper` 和 `right_gripper` 控制夹爪，值为 `open`
或 `close`。省略这两个字段时，该阶段不会发送夹爪命令。

流程文件采用 JSON 兼容的 YAML 写法，因此不需要为执行器额外安装 PyYAML。
