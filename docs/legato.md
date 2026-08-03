# Legato 使用说明

本仓库把 Legato 做成 Pi0/Pi0.5 的可选训练和推理方式。默认不开启时，原来的 flow matching loss，以及 `sync_chunk`、`async_queue`、`rtc_guidance` 执行方式都不变。

## 新增内容

模型侧通过 `Pi0Config.legato_enabled=True` 开启 Legato。开启后，训练会使用 Legato velocity target，并把 schedule 值 `omega` 写入一个空闲 action 维度。这样不会改变模型参数 shape，因此原来的 Pi0/Pi0.5 预训练 checkpoint 仍然可以加载。

机器人执行侧使用独立模式：

```yaml
execution_mode: legato
```

这会启动 `robot_client/runners/legato.py`，不是原来的 RTC guidance runner。

## 训练入口

Legato 训练仍然使用原来的 JAX 训练脚本：

```bash
python scripts/train.py <config_name> --exp-name <run_name>
```

NZ100 相关训练配置在 `src/openpi/training/config.py` 中：

```text
pi05_nz100
pi05_nz100_lora
```

Legato 参数属于 `Pi0Config`，有两种传入方式：

1. 直接修改 `src/openpi/training/config.py` 中对应 `TrainConfig` 的 `model=pi0_config.Pi0Config(...)`；
2. 启动 `scripts/train.py` 时通过 `--model.legato-*` 参数覆盖。

## 推荐 NZ100 训练命令

全量微调示例：

```bash
python scripts/train.py pi05_nz100 \
  --data.repo-id /path/to/lerobot_dataset \
  --assets-base-dir /path/to/openpi_assets \
  --checkpoint-base-dir /path/to/openpi_checkpoints \
  --exp-name nz100_legato \
  --num-train-steps 30000 \
  --batch-size 32 \
  --num-workers 8 \
  --save-interval 1000 \
  --keep-period 5000 \
  --fsdp-devices 1 \
  --model.legato-enabled \
  --model.legato-omega-dim 31 \
  --model.legato-loss-action-dim 16 \
  --model.legato-train-num-steps 10 \
  --model.legato-full-guidance-steps 5 \
  --model.legato-ramp-steps 15
```

如果使用 `scripts/train_nz100.sh`，把这些 `--model.legato-*` 参数加到最后的 `python scripts/train.py pi05_nz100 ...` 命令后面即可。

## 随机 Schedule 训练

如果希望训练时覆盖不同延迟 `d` 和不同 ramp 长度 `r`，可以开启随机 schedule：

```bash
--model.legato-enabled \
--model.legato-omega-dim 31 \
--model.legato-loss-action-dim 16 \
--model.legato-train-num-steps 10 \
--model.legato-randomize-schedule \
--model.legato-full-guidance-min 0 \
--model.legato-full-guidance-max 8 \
--model.legato-ramp-min 0 \
--model.legato-ramp-max 20
```

当前实现中，`d` 和 `r` 会在对应范围内按整数均匀采样。

## 模型参数说明

`legato_enabled`
: 是否启用 Legato loss 分支。默认是 `False`。

`legato_omega_dim`
: 用来承载 `omega` 的空闲 action 维度。以 NZ100 为例，如果模型 action dim 是 32，而真实 raw action 是 16 维，通常可以选择 `31`。使用前需要确认这个维度不会被 action transform 或机器人输出使用。

`legato_loss_action_dim`
: 参与 loss 的真实 action 前缀维度数。NZ100 raw action chunk 通常是 `16`。`legato_omega_dim` 总会从 loss 中排除。

`legato_train_num_steps`
: 构造 Legato training target 时使用的 denoise 步数。建议尽量和推理时 `num_steps` 保持一致。

`legato_full_guidance_steps`
: 固定 schedule 训练时的 `d`，表示 chunk 前多少步使用 full continuation guidance。

`legato_ramp_steps`
: 固定 schedule 训练时的 `r`，表示 full guidance 后多少步线性衰减到 0。

`legato_randomize_schedule`
: 开启后，每个训练 sample 会从配置范围中随机采样 `d` 和 `r`。

## 训练公式

OpenPI 代码里使用的时间方向是：

```text
t=1: noise
t=0: action
```

因此当前实现中的 Legato loss 是论文公式的时间反转版本：

```text
x_t = t * eps + (1 - t) * A
guided_x_t = (1 - omega) * x_t + omega * A
kappa = omega / dt
u_legato = (1 - kappa * t) * (eps - A)
loss = MSE(v_theta(guided_x_t, observation, t, omega), u_legato)
```

其中：

```text
omega = 1 表示完全锚定 action/reference prefix
omega = 0 表示自由生成
```

## 和 Legato-Kinetix 原仓库的对应关系

`/home/pc/VLA/Legato-kinetix` 使用的是 `warmup_min`、`warmup_max`、`warmup_sampling`。它训练时采样的是 hard switch schedule：

```text
i < warmup   -> 锁定前缀
i >= warmup  -> 自由生成
```

换成当前仓库的 `omega` 方向，等价于：

```text
d = warmup
r = 0
```

如果想模拟 Legato-kinetix 原仓库的 hard switch 行为，可以使用：

```bash
--model.legato-randomize-schedule \
--model.legato-full-guidance-min 0 \
--model.legato-full-guidance-max 4 \
--model.legato-ramp-min 0 \
--model.legato-ramp-max 0
```

论文版本包含 ramp，因此如果要更接近论文，应使用非零的 `legato_ramp_steps` 或 `legato_ramp-max`。

## 推理配置

机器人端 Legato 参数在 `robot_client/configs/nz100_client.yaml` 中：

```yaml
execution_mode: legato
legato_execute_horizon: 35
legato_prefix_len: 5
legato_ramp_end: 20
legato_delay_buffer_size: 4
legato_max_delay_steps: 12
```

`legato_prefix_len`
: full-guidance prefix 的最小长度。运行时它也是动态估计延迟 `d` 的下限。

`legato_ramp_end`
: ramp 区域结束位置。如果 `d=5` 且 `legato_ramp_end=20`，那么 `r=15`。

`legato_delay_buffer_size`
: 保存最近多少次推理延迟。runner 会取这个最近窗口里的最大值作为下一次延迟估计。

`legato_max_delay_steps`
: 动态延迟估计 `d` 的上限，防止一次异常慢推理导致前缀锁定过长。

## 启动 Legato 推理

先像平时一样启动 policy server，并加载 Legato 训练出来的 checkpoint。然后运行：

```bash
python -m robot_client.main --config robot_client/configs/nz100_client.yaml
```

如果只是做连通性或 mock 检查：

```bash
python -m robot_client.main --config robot_client/configs/nz100_client.yaml --execution-mode legato --mock --once
```

## 注意事项

目前 Legato 只实现了 JAX Pi0/Pi0.5 路径，PyTorch 训练和推理还没有实现。

如果使用 LoRA 训练，需要确认模型有足够的可训练参数去学习 `omega` carrier 信号。第一轮验证建议优先使用全量微调。
