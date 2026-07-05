# Experiments

本文件记录已确认的训练、评估和计划实验。任何结果必须来自实际日志、CSV、checkpoint、配置文件或明确人工记录；无法确认的信息写“待确认”。

## 记录规则

- 每个实验必须有唯一 ID。
- 必须记录 task id、场景、物体、策略、配置、checkpoint、日志目录、seed、评估命令和结果来源。
- 不得根据目录名臆造成功率、训练轮数、最佳 checkpoint 来源或结论。
- 若只确认 artifact 存在，但未计算汇总指标，应明确写“汇总结果待确认”。

## EXP-003 — GelFusion-style RL comparison plan

- 状态：计划中；代码和配置已新增，训练/评估结果待确认
- 实验类型：视触融合方法对比
- 论文参考：Jiang et al. 2025 GelFusion 思路；当前实现是 PPO/RL 特征级复现，不是 Diffusion Policy 原版。
- Task id：`TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube`
- Scene：`Drawer-Occlusion`
- Object：`Cube`
- Policy / method：GelFusion-style vision-led cross attention + tactile dynamic stats
- Agent 配置：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_gelfusion.yaml`
- Env / actor 代码：
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_box.py`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_policy.py`
- Seed：42，依据 `ppo_vt_gelfusion.yaml`
- Observation：
  - `third_resnet: 256`
  - 四路 `tactile_*_depth_resnet: 256`
  - `tactile_dynamic_stats: 8`
  - `proprio_obs: 18`
  - critic privileged keys 沿用 VT 环境
- Action space：5，沿用 `[dx, dy, dz, dyaw, gripper]`
- Checkpoint：待训练产生
- 日志目录：待训练产生

### EXP-003 推荐训练命令

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras
```

### EXP-003 推荐对比对象

- 视觉退化 baseline：`TacEx-V-Downsample-Drawer-Occlusion-Cube`
- 普通 VT baseline：`TacEx-VT-Downsample-Drawer-Occlusion-Cube`
- 现有 cross/aux 分支：`TacEx-Tactile-Cross-Downsample-Drawer-Occlusion-Cube` 或 `TacEx-Tactile-Cross-Alpha-Aux-Downsample-Drawer-Occlusion-Cube`

### EXP-003 推荐评估命令

```bash
python scripts/reinforcement_learning/skrl/play_bucket.py \
  --task TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube \
  --checkpoint <gelfusion_run>/checkpoints/best_agent.pt \
  --num_envs 16 \
  --enable_cameras \
  --bucket_rounds 1 \
  --headless
```

### EXP-003 结果

| 指标 | 值 | 状态 |
| --- | --- | --- |
| 普通评估 recent success rate | 待确认 | 未训练 |
| Bucket 总成功率 | 待确认 | 未训练 |
| 按遮挡程度分组成功率 | 待确认 | 未训练 |
| 与 VT-Downsample 差异 | 待确认 | 未训练 |

## EXP-001 — VT Downsample Drawer-Occlusion Cube artifact

- 状态：已完成 artifact 存在确认；实验结论待确认
- 实验类型：训练后普通评估与 bucket 评估 artifact 记录
- 日志目录：`logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/`
- 日志目录名时间：`2026-05-30_21-33-24`；实际训练开始/结束时间待确认
- Git commit：待确认
- Git branch：待确认
- Task id：`TacEx-VT-Downsample-Drawer-Occlusion-Cube`，已在当前文档评估命令中使用；具体训练 task id 是否完全相同待确认
- Scene：`Drawer-Occlusion`
- Object：`Cube`
- Policy / method：VT Downsample，具体 class 从该 run 的 `params/agent.yaml` 确认
- Agent 配置 artifact：`logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/params/agent.yaml`
- Env 配置 artifact：`logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/params/env.yaml`
- Checkpoint：`logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt`
- Seed：42，依据该 run 的 `params/agent.yaml`
- Num envs：128，依据该 run 的 `params/env.yaml`

### EXP-001 观测与网络

- Action space：5，依据该 run 的 `params/env.yaml` 和 `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.action_space`
- 视觉 key：`third_resnet`
- 触觉 keys：
  - `tactile_left_depth_resnet`
  - `tactile_right_depth_resnet`
  - `tactile_left_down_depth_resnet`
  - `tactile_right_down_depth_resnet`
- Critic 输入：privileged state，依据该 run 的 `params/agent.yaml`
- 网络层：`[512, 256, 128, 64]`，依据该 run 的 `params/agent.yaml`

### EXP-001 PPO 超参数

| 字段 | 值 | 来源 |
| --- | --- | --- |
| seed | 42 | `params/agent.yaml` |
| rollouts | 128 | `params/agent.yaml` |
| learning_epochs | 4 | `params/agent.yaml` |
| mini_batches | 16 | `params/agent.yaml` |
| discount_factor | 0.99 | `params/agent.yaml` |
| lambda | 0.95 | `params/agent.yaml` |
| learning_rate | 5e-4 | `params/agent.yaml` |
| entropy_loss_scale | 0.05 | `params/agent.yaml` |
| value_loss_scale | 1.0 | `params/agent.yaml` |
| timesteps | 500000 | `params/agent.yaml` |

### EXP-001 评估 artifact

普通评估 metrics：

```text
logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/metrics/play/play_metrics_20260531_131229.csv
```

已确认前几行包含字段：

```text
step,recent_success_rate,window_success_rate
200,0.625,0.625
400,0.59375,0.5625
600,0.6354166865348816,0.71875
```

Bucket trial metrics：

```text
logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/metrics/play_bucket/play_bucket_20260531_152044.csv
```

Bucket summary metrics：

```text
logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/metrics/play_bucket/play_bucket_20260531_152044_summary.csv
```

已确认 summary 字段：

```text
grid_index,x,y,trials,successes,success_rate,occlusion_ratio_mean
```

已确认现象：该 summary 中存在 `occlusion_ratio_mean` 为 `nan` 的记录；原因与 bbox/occlusion 设置的关系待确认，代码层风险见 `docs/KNOWN_ISSUES.md:ISSUE-008`。

### EXP-001 评估命令

普通评估命令：

```bash
python scripts/reinforcement_learning/skrl/play.py --task TacEx-VT-Downsample-Drawer-Occlusion-Cube --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt --num_envs 16 --enable_cameras
```

Bucket 评估命令：

```bash
python scripts/reinforcement_learning/skrl/play_bucket.py --task TacEx-VT-Downsample-Drawer-Occlusion-Cube --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt --num_envs 16 --enable_cameras
```

### EXP-001 结果

| 指标 | 值 | 状态 |
| --- | --- | --- |
| 普通评估 recent success rate | 见 `play_metrics_20260531_131229.csv` | 已有 CSV，未在本文档汇总最终值 |
| 普通评估 window success rate | 见 `play_metrics_20260531_131229.csv` | 已有 CSV，未在本文档汇总最终值 |
| Bucket 总成功率 | 待确认 | 需要从 trial 或 summary CSV 计算 |
| 按遮挡程度分组成功率 | 待确认 | 当前已确认 summary 中 `occlusion_ratio_mean` 可为 `nan` |
| 平均回报 | 待确认 | 当前未确认对应日志字段 |

## EXP-002 — UR10 Robotiq 2F85 pick-place skrl checkpoint artifact

- 状态：已完成 artifact 迁移；训练效果待确认
- 实验类型：本机 IsaacLab skrl 训练 artifact 迁移记录
- 原始来源：`/home/tinydog/IsaacLab/logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/`
- 迁移后日志目录：`logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/`
- Task id：大概率为 `Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0`，依据 `params/env.yaml` 中 2F85 USD 和环境配置；原始 run 未直接记录 task id，待确认。
- Agent 配置 artifact：`logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/params/agent.yaml`
- Env 配置 artifact：`logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/params/env.yaml`
- Checkpoint：
  - `logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/checkpoints/best_agent.pt`
  - `logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/checkpoints/agent_20000.pt`
- Seed：42，依据 `params/agent.yaml`
- Num envs：1024，依据 `params/env.yaml`
- Observation space：27，依据 `params/env.yaml`
- Action space：4，依据 `params/env.yaml`
- Robot USD：`source/tacex_assets/tacex_assets/data/Robots/URRobotiq/ur10_robotiq_2f85.usda`

### EXP-002 评估命令

```bash
python scripts/reinforcement_learning/skrl/play.py --task Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0 --checkpoint logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/checkpoints/best_agent.pt --num_envs 16
```

### EXP-002 结果

| 指标 | 值 | 状态 |
| --- | --- | --- |
| Isaac Sim 回放 | 待确认 | 本次只迁移 artifact，未运行 play |
| 抓取/放置成功率 | 待确认 | 未找到评估 CSV |
| 平均回报 | 待确认 | 未汇总 TensorBoard event |

## Planned Experiments

当前无已从仓库确认的计划实验。新增计划实验时使用以下模板。

```text
## EXP-XXX — 标题

- 状态：计划 / 进行中 / 已完成 / 已废弃
- 目标：
- Git commit：
- Git branch：
- Task id：
- Scene：
- Object：
- Policy / method：
- Agent config：
- Env config：
- Seed：
- Num envs：
- Checkpoint：
- 日志目录：
- 训练命令：
- 评估命令：
- 主要变量：
- 预期对比：
- 结果来源：
- 结果：
- 结论：
- 待确认：
```
