# NZ100 trajectory replay

最小轨迹回放工具：从 LeRobot episode 的 `action` 字段读取 16 维动作，并通过当前 `robot_client` 的 ROS2 关节轨迹 + YSRobot Modbus 夹爪接口回放。

默认只 dry-run，不会控制机器人：

```bash
cd /home/pc/VLA/openpi
python3 -m replay.replay_episode --episode 0
```

真正执行：

```bash
cd /home/pc/VLA/openpi
source /opt/ros/humble/setup.bash
python3 -m replay.replay_episode \
  --dataset /home/pc/VLA/lerobot_data_collection/data_lerobot \
  --episode 0 \
  --config robot_client/configs/nz100_client.yaml \
  --execute
```

常用参数：

```bash
# 只回放前 100 步
python3 -m replay.replay_episode --episode 0 --max-steps 100 --execute

# 不执行启动回位
python3 -m replay.replay_episode --episode 0 --no-home --execute

# 手动指定回放频率
python3 -m replay.replay_episode --episode 0 --fps 30 --execute

# 只回放片段 [50, 150)
python3 -m replay.replay_episode --episode 0 --start 50 --end 150 --execute
```

动作顺序必须是当前 NZ100 16 维格式：

```text
0:7     左臂 7 个关节
7       左夹爪，1=开，2=关
8:15    右臂 7 个关节
15      右夹爪，1=开，2=关
```
