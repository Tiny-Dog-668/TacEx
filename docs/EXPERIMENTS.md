# Experiments

本文件记录已确认的训练、评估和计划实验。任何结果必须来自实际日志、CSV、checkpoint、配置文件或明确人工记录；无法确认的信息写“待确认”。

## 记录规则

- 每个实验必须有唯一 ID。
- 必须记录 task id、场景、物体、策略、配置、checkpoint、日志目录、seed、评估命令和结果来源。
- 不得根据目录名臆造成功率、训练轮数、最佳 checkpoint 来源或结论。
- 若只确认 artifact 存在，但未计算汇总指标，应明确写“汇总结果待确认”。

## EXP-004 — Real-reference-aligned vision-only Cube PPO

- 状态：历史 200000-step checkpoint artifact 和 TorchScript export 已存在，但它们来自 2026-07-12 strict sim2real contract 修复之前，仅保留作历史记录；修复后策略待重新训练
- 实验类型：Franka + D435 vision-only Cube sim2real
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-v0`
- 现实依据：`20260711_214450_real_alignment_reference/alignment_reference.json` 和同目录 RGB reference images
- 环境：`source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_real_alignment_env.py`
- 当前 Agent：`source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_real_alignment_cfg_resnet18.yaml`，Actor mean 已改为 `tanh(ACTIONS)`
- 随机化 profile：Clean；只随机 cube XY reset，关闭图像、光照、地面和台面颜色随机化
- Seed：42
- Actor observation：`wrist_resnet:512`、`proprio_obs:15`、`action_history:4`
- Action：4 维 `[dx, dy, dz, gripper_total_width_delta]`；当前环境逐维物理 command scale 为 `[0.05,0.05,0.05,0.01] m/policy step`。RMA checkpoint 内 action-history normalizer 仍为 `[0.025,0.025,0.025,0.005]`，必须存储实际下发的物理增量后再由模型按其训练尺度归一化；历史 v8/v7 分别为 25/2 mm 与 10/2 mm。
- Object：实测尺寸 `0.05x0.05x0.05 m`
- Object reset：Franka base/world 对齐坐标以 `(0.50,0.00,0.026) m` 为中心；前 20k policy steps 使用 `x=[0.48,0.52], y=[-0.02,0.02] m`，20k–100k 线性扩展，之后为完整 `x=[0.45,0.55], y=[-0.05,0.05] m`。play、bucket 和 rollout 强制完整范围
- Geometry：台面厚度 `1 mm`、顶面 `z=0.001 m`，5 cm 方块初始质心 `z=0.026 m`
- Reward：`5*reach + 15*lift + 100*success + table_collision_penalty`；质心抬升 0–35 mm 线性映射到 lift `[0,1]`，35 mm 连续保持 5 个 policy steps 触发 success；倾角只用于诊断，不参与 reward、success 或 done。目标机械臂 link 对台面最大接触力 `>1 N` 时该 step 加 `-10`
- Frequency：physics 60 Hz；camera/render/policy/action/observation/reward/history 30 Hz（`sim.dt=1/60, decimation=2`）
- Episode：5 秒，150 个 policy steps
- 当前策略 camera resolution/intrinsics：`224x224`，有效 `K=[338.742544,0,123.748857; 0,340.550811,120.393372; 0,0,1]`。它对应现实 `640x480` 裁剪 `x=[100,500), y=[34,432)` 后将 400x398 双线性缩放到 224x224；仿真用 centered 300 px-focal coverage render 加固定 GPU warp 实现该有效 K
- 当前 Panda base world pose：`pos=(0,0,0.02) m`，以匹配真机 20 mm 垫高；桌面/方块仍在 world `z=0` 参考下。`base_T_camera_color_optical` 的 `pos=(1.166091088407,0.035901608197,0.514200335898)`、Isaac `rot(wxyz)=(0.378248136306,-0.604227000834,-0.586824979121,0.384024117374)`、`convention=ros`；相机作为独立 world prim 的实际 position 为 `(1.166091088407,0.035901608197,0.534200335898)`，以保持该 base 外参。部署相机序列号锁定为 `215322076207`。
- Num envs：256，依据 `training_summary.txt`
- 日志目录：`logs/skrl/sim2real_cube_real_alignment/2026-07-11_23-15-23_ppo_torch_vision_only_resnet18/`
- Checkpoint：`checkpoints/best_agent.pt`；同目录存在 `agent_200000.pt`
- 历史部署 export：`checkpoints/exported/policy_actor_e2e_best_agent.pt` 和同名 JSON metadata；不能作为修复后 contract 的部署文件
- 历史 Export contract：raw uint8 RGB `[N,224,224,3]`、`proprio_obs [N,15]`、`action_history [N,4]` -> unbounded raw mean action `[N,4]`，nominal frequency 30 Hz
- Export validation：trace max abs error 0；synthetic/sim start/real median RGB 均输出 finite `(1,4)`，同输入重复误差 0；本机 CPU 平均推理约 8.4-9.3 ms
- 普通 play metrics：`metrics/play/play_metrics_20260712_120929.csv` 当前记录的 success rate 为 `nan`，不能据此报告成功率
- Gate-removal 前最后一轮训练：`logs/skrl/sim2real_cube_real_alignment/2026-07-25_22-01-32_ppo_torch_vision_only_resnet18/` 完成 200k steps，manifest 为 contract v3、40°→10° success curriculum，但仍声明 `privileged_dz_gate=applied_after_action_history_and_not_available_on_the_real_robot`；其 `best_agent.pt`、`agent_200000.pt` 不兼容当前 v9，无论训练标量如何都不得续训、导出或部署。对应 play CSV 的 success rate仍为 `nan`。
- 已确认 smoke：环境可创建、reset、执行零动作；观测 shape 和动作空间正确。
- 旧奖励 smoke（已被当前质心语义替代）：reset 后最低角点日志曾为 `lift_delta≈0.001 m`。当前边界测试要求质心 delta `[0,0.0175,0.035] m` 得到 lift `[0,0.5,1]` 和 success `[False,False,True]`。
- 物理可抓取证据（不是当前 v9 策略效果）：`logs/skrl/sim2real_cube_real_alignment/2026-07-25_20-43-05_ppo_torch_vision_only_resnet18/` 的 512-env 旧奖励训练日志在 step 4000 记录 `Instantaneous reward (max)=119.78437`，并在 step 4500 记录非零 `reward/success=0.000148437495`。这确认旧场景至少出现过完整 lift+success 的物理轨迹，但不能据此给出当前策略成功率。
- Reach 几何隔离测试（2026-07-30，4-env Isaac）：reset 时左右指尖中心分别约为 `(0.4999,-0.0200,0.2995)` 与 `(0.4998,0.0200,0.2995) m`，中点约为 `(0.49984,0.000043,0.29952) m`；该中点与 `panda_hand+0.1034 m` IK TCP 的误差最大约 `1.37e-7 m`。使用生产 `_get_rewards()` 且隔离其他奖励后，方块质心位于中点、偏移 25 mm、偏移 50 mm、位于 `panda_hand` origin 时，reach 分别为 `1.0000/0.7551/0.5379/0.2245`，确认峰值位于实际指尖中点而非 hand origin。当前 reset 距离约 `0.274 m`，在 `reach_sigma=0.1 m` 下 raw reach 仅约 `0.0083`，因此早期 reach 数值小主要来自长距离 tanh 饱和，不是中心坐标错误。
- 当前训练监控：`episode_success_rate_window` 是最近 200 个 policy steps 内成功结束 episode 数 / 所有结束 episode 数；一个并行 step 可贡献多个完成 episode，无完成 episode 的 step 仍推进窗口。日志同时记录窗口成功/完成数、累计成功/完成数和累计成功率；它与单步 `reward/success` 分开，后者仍是当前达到 35 mm 的环境比例。
- 历史图像对齐 smoke：旧相机 world `pos=(1.90,0.0,0.468)`、向下约 15° 时，seed 42 仿真帧 RGB mean `[70.6,113.0,70.1]`，现实 model median 为 `[80.1,113.0,81.9]`；方块中心分别为 `(86.6,128.6)` 和 `(85.6,128.5)` pixel。这是旧位姿下的单帧外观检查，不适用于当前相机位姿，也不是策略结果。
- 当前训练 contract v9：ResNet18 始终 `eval()` 且参数冻结；Clean/DR `action_history` 为上一拍逐维 scaled/clipped requested physical command `[0.05*u_x,0.05*u_y,0.05*u_z,0.01*g]`。RMA Student 训练/部署模型对该 history 使用 checkpoint 固有 `[0.025,0.025,0.025,0.005]` normalizer，二者不能只改其一。夹爪目标宽度每个 policy step 只累加一次，physics application 只重发缓存目标；IK TCP 为固定 `panda_hand+0.1034 m`，reach/critic 使用左右指尖世界坐标中点；Actor 后控制链不读取 ground-truth object XY；Actor deterministic mean 为 `tanh(raw_mean)`。Contract 同时记录新 crop/有效 K/GPU 补偿、相机序列号、`x/y ±5 cm` 物体课程、台面、无倾角 success 和碰撞奖励语义。
- Checkpoint 兼容性：上述历史 checkpoint 虽然张量 shape 不变，但已经适应旧 BatchNorm、两拍延迟 history、unbounded mean、per-finger/substep 夹爪语义、旧抓取中心或不同 policy frequency，不能在当前环境中 resume 后继续训练；旧 artifact 只能按其保存的环境配置复现，需要按当前 30 Hz/5 秒配置从头重新训练。
- 当前短训练 run：`logs/skrl/sim2real_cube_real_alignment/2026-07-26_20-51-57_ppo_torch_vision_only_resnet18/`；`best_agent.pt` SHA-256 为 `92c930d2605b8abf58466cbb4717f2004bd92a96629937f62c05efae7fc341d3`。截至约 11.5k steps，TensorBoard 的 reach reward 从约 `0.023` 上升、最好约 `0.331`，reach distance 从约 `0.234 m` 降至约 `0.120 m`；lift 仅短暂约 `0.014`，success 近似为 0。因此它只能用于受控真机链路诊断，不能视为已学会抓取。
- 当前 v8 export：`checkpoints/exported/policy_actor_e2e_best_agent.pt`，SHA-256 `b888e7321c0481f80623ffa84837f32bc8f426eefbd1af33937764d787744c24`；复制到 `franka/checkpoint/real_alignment_v8_25mm/exported/`，独立 JIT 测试确认输出 finite、bounded、repeat diff 为 0。真机成功率：待确认。
- 修复后链路 smoke：`logs/skrl/sim2real_cube_real_alignment/2026-07-12_14-29-22_ppo_torch_vision_only_resnet18/` 以 1 env 完成 128 steps/一次 PPO update，并用测试 `agent_128.pt` 跑通 RGB/feature export、独立 JIT tester 和 2-step deterministic play。该 checkpoint 只验证代码链路，不计作策略效果实验。
- 待确认：新 D435 外参的标定误差、新 crop 下当前 `x=[0.45,0.55] m` 全范围方块的完整可见性、真实方块质量/摩擦、真机实际 30 Hz 抖动和成功率。当前 v9 尚无新训练 checkpoint 或真机 bundle；历史 v8 仿真与独立真机配置仍声明 25/2 mm，不得用于 v9 环境。

### EXP-004 推荐训练命令

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-v0 \
  --num_envs 256 \
  --enable_cameras \
  --start_frame_count 5 \
  --headless
```

训练完成后在完整位置范围评估至少 100 个完整 episode；当前接受标准为 success rate
`>=80%` 且机械臂撞桌 transition 比例 `<1%`。达到标准前不导出真机 bundle。

## EXP-004-DR — Real-reference-aligned broad domain-randomized Cube PPO

- 状态：延后 DR 的 300k curriculum 环境和训练配置已实现，待从头训练；旧 200k DR run 不等价于当前 profile
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-DR-v0`
- 继承：与 EXP-004 Clean 使用相同机器人、相机、方块、动作、观测、奖励和 done
- 课程：0–100k outer policy step 使用 `scale=0` 的精确 Clean 视觉场景，并先完成 20k–100k 物体位置课程；100k–220k 从 0 线性增至 100%；220k–300k 保持 full range。`common_step_counter` 与 `num_envs` 无关，当前 scale 记录为 `info/dr_curriculum_scale`，resume offset 当前需手动设置。
- 相机 DR：以标定位姿为中心，full range 为 XYZ 各 `±3 mm`、RPY 各 `±1°`；GPU 等效 focal scale `[0.985,1.015]`、principal point shift 各 `±2 px`
- 图像 DR：brightness/contrast/saturation/gamma 均 `[0.85,1.15]`、hue `±5°`、white-balance red/blue shift `±0.08`、Gaussian blur probability `0.15` / kernel `3`、Gaussian noise std `[0,0.01]`
- 场景 DR：plate 以 Clean `(0.02,0.02,0.02)` 为中心并在 full scale 取 `[0.01,0.05]`，backdrop 以 Clean `(0.01,0.01,0.01)` 为中心并取 `[0.005,0.03]`；batch-global DomeLight 以 Clean intensity/color 为中心，full range intensity `[1000,3000]`、relative color-temperature tint `[3800,7200] K`；global ground 固定
- 未包含：质量、摩擦和机器人动力学随机化
- Agent：`source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_real_alignment_dr_cfg_resnet18.yaml`
- 日志目录：`logs/skrl/sim2real_cube_real_alignment_dr/`
- Trainer：300000 outer policy steps
- checkpoint、训练成功率和真机成功率：待确认

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-DR-v0 \
  --num_envs 4 \
  --enable_cameras \
  --headless
```

## EXP-004-P — Real-Alignment privileged-position reward/control upper bound

- 状态：环境和 Agent 配置已实现；训练与成功率待确认
- 目标：去除视觉定位变量，判断当前动作、物理和 reach/lift/success 奖励能否学会稳定抓取
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-Privileged-v0`
- Environment：`source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_real_alignment_privileged_env.py`
- Agent：`source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_real_alignment_privileged_cfg.yaml`
- Actor observation：28-D，`proprio_obs:15 + action_history:4 + privileged_cube_pos:3 + privileged_gripper_pos:3 + privileged_target_pos:3`
- Position frame：去除 `scene.env_origins` 的 robot-root-aligned per-env frame
- Camera/encoder：均不创建；无需 `--enable_cameras`
- Action/reward/done：继承 EXP-004 Clean，保持 4-D total-width action、质心 0–35 mm lift、无倾角 gate、5-step hold、撞桌 `-10` 和 150-step horizon
- PPO：与 Clean baseline 保持 rollouts 128、4 epochs、16 minibatches、learning rate `3e-4`，用于尽量只比较观测差异
- 日志目录：`logs/skrl/sim2real_cube_real_alignment_privileged/`
- Checkpoint、训练成功率、确定性 play 成功率：待确认
- 训练链路 smoke：`logs/skrl/sim2real_cube_real_alignment_privileged/2026-07-26_16-18-10_ppo_torch_privileged_position/` 使用 4 env 完成 128 steps/一次 PPO update；只证明环境、28-D Actor、49-D Critic 和 Runner 接通，不计作抓取效果。
- 部署：不允许；Actor 依赖 simulator ground truth

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-Privileged-v0 \
  --num_envs 256 \
  --headless
```

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

## EXP-003 — Real-Alignment v3 exported Actor 批量 rollout 诊断

- 状态：已完成诊断采集；该 checkpoint 不满足当前 v9 部署 contract
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-v0`
- Policy / method：PPO vision-only ResNet18 exported deterministic Actor
- Seed：42
- Num envs：32
- Policy steps：450（每个 episode 最多 150）
- Checkpoint：`logs/skrl/sim2real_cube_real_alignment/2026-07-25_22-01-32_ppo_torch_vision_only_resnet18/checkpoints/best_agent.pt`
- Exported Actor：`logs/skrl/sim2real_cube_real_alignment/2026-07-25_22-01-32_ppo_torch_vision_only_resnet18/checkpoints/exported/policy_actor_e2e_best_agent.pt`
- Contract：v3；显式 legacy diagnostic，仅复现保存的 `env.pkl`，不表示当前 v9/真机兼容
- 结果来源：
  - `metrics/sim2real_rollouts/rollout_v3_32env_450steps.npz`
  - `metrics/sim2real_rollouts/rollout_v3_32env_450steps.summary.json`
  - `metrics/sim2real_rollouts/rollout_v3_32env_450steps.analysis.json`
  - `metrics/sim2real_rollouts/rollout_v3_32env_450steps.actions.csv`

### EXP-003 结果

| 指标 | 值 |
| --- | ---: |
| Transition 数 | 14,400 |
| 完成 episode | 98 |
| Success terminal | 7 |
| 完成 episode 成功率 | 7.14% |
| episode 第一步 mean action `[x,y,z,g]` | `[0.583,-0.307,-0.960,0.765]` |
| episode 第一步 Z 近饱和比例 | 93.08% |
| episode 第一步 gripper 近饱和比例 | 89.23% |
| 全部 transition gripper 近饱和比例 | 51.36% |
| 夹爪目标位于 80 mm 上限比例 | 76.51% |
| legacy privileged dz gate 触发比例 | 0.00% |
| 最大质心相对抬升 | 59.10 mm |

结论：该策略在仿真中已经不是可靠抓取策略。前 30 步主要输出接近最大幅度的 XYZ 运动，并持续给出正夹爪增量（张开），绝大部分时间把夹爪目标推到 80 mm；因此真机表现差不能只归因于视觉 sim-to-real gap。真机 0712 配置的 XYZ/夹爪 scale、history scale/delay 又与 v3 metadata 不一致，会进一步改变闭环行为。

## Planned Experiments

## EXP-005 — Real-Alignment RMA 教师学生蒸馏

- 状态：计划；代码路径已实现，训练未执行
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0` / `...-Student-v0`
- Scene：Real-Alignment Clean v9
- Policy / method：200k privileged PPO Teacher + 100k single-frame visual position/contact/action/action-rate distillation
- Teacher Actor external input：`proprio[15] + history[4] + cube_xyz_root[3] + left_right_contact[2]`
- Teacher Actor network feature：上述24维 + `gripper_xyz_root_from_fk[3] + (cube-gripper)_xyz[3]` = 30维；不使用物体/末端 quaternion
- Student deployment input：`RGB[224,224,3] + proprio[15] + history[4]`
- 评估：固定10x10 XY 网格，共100 episodes
- RMA-only reward：0.2 N接触阈值；单侧接触奖励3.0/step；动作变化惩罚 `-0.05 * mean(((a_t-a_{t-1}) / action_scale)^2)`
- RMA-only gripper actuator：XYZ动作尺度为50 mm/step，夹爪总宽度动作尺度为10 mm/step；finger `effort_limit_sim=40`、`stiffness=400`、`damping=40`
- Done：RMA success 不终止 episode；仅 timeout 或严重机器人穿地碰撞终止，episode 成功统计按曾经达到 success 计算
- 验收：Teacher success >=80%；Student success >=0.9 Teacher；Student 3D RMSE <=15 mm
- checkpoint：RMA Teacher manifest v6 及更早 artifact 已失效；当前 v7 需从头训练 Teacher 后再蒸馏 Student
- 结果：待训练与评估，当前不得写为已达标

新增计划实验时使用以下模板。

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
